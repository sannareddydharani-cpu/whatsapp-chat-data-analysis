import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import StringIO
import warnings
import plotly.io as pio
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import emoji
from typing import List, Dict, Any

# --- Configuration and Setup ---
# FIX: Using a string-based filter to avoid the pandas AttributeError on import.
warnings.filterwarnings('ignore', category=UserWarning, message='A value is trying to be set on a copy of a slice from a DataFrame')
pio.templates.default = "plotly_white"

# --- Constants ---
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
WA_MESSAGE_START_PATTERN = re.compile(
    r'^\[(\d{2}/\d{2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2})\s*(\u202f|)(AM|PM|am|pm|)\s*\]\s*([^:]+):\s*(.*)',
    re.MULTILINE | re.IGNORECASE
)

CUSTOM_STOPWORDS = set(STOPWORDS)
CUSTOM_STOPWORDS.update([
    'media', 'omitted', 'message', 'deleted', 'pm', 'am', 'hai', 'kya', 'bhai', 'bhi', 'hum', 'aur', 
    'sir', 'guys', 'apni', 'apne', 'sath', 'jo', 'woh', 'yeh', 'kon', 'kar', 'karke', 'kiya', 'bhi', 
    'liye', 'ko', 'per', 'pe', 'hai', 'nahi', 'par', 'from', 'to', 'for', 'the', 'is', 'it', 'and', 
    'in', 'of', 'i', 'you', 'me', 'will', 'was', 'we', 'are', 'was', 'were', 'have', 'had', 'has', 
    'do', 'did', 'does', 'can', 'could', 'would', 'should', 'get', 'like', 'with', 'about', 'just',
    'one', 'only', 'may', 'must', 'ask', 'them', 'this', 'that', 'plz', 'please', 'aap', 'sir', 
    'mam', 'b', 'c', 'msg', 'whatsapp', 'messages', 'call'
])


# --- 1. Data Processing Logic (Consolidated) ---

class WhatsAppDataProcessor:
    def __init__(self, uploaded_file: StringIO):
        self.file_content = uploaded_file.getvalue()
        self.df = pd.DataFrame()

    def parse_chat(self) -> None:
        data: List[Dict[str, Any]] = []
        current_message: Dict[str, Any] = {}
        lines = self.file_content.splitlines()

        for line in lines:
            match = WA_MESSAGE_START_PATTERN.match(line)

            if match:
                if current_message:
                    data.append(current_message)

                date_part = match.group(1)
                time_ampm_part = f"{match.group(3)}{match.group(4).strip()}" if match.group(4) else ""
                time_part = f"{match.group(2)}{time_ampm_part}"

                current_message = {
                    'DateTimeStr': f"{date_part} {time_part}",
                    'Sender': match.group(5).strip(),
                    'Message': match.group(6).strip()
                }
            else:
                if current_message:
                    current_message['Message'] += '\n' + line.strip()
                continue

        if current_message:
            data.append(current_message)

        self.df = pd.DataFrame(data)

    def process_dataframe(self) -> pd.DataFrame:
        if self.df.empty:
            return self.df

        # 1. Clean and Prepare
        self.df['Message'] = self.df['Message'].str.replace(r'^\u200e|\u200f', '', regex=True).str.strip()
        self.df.dropna(subset=['Message'], inplace=True)
        self.df = self.df[self.df['Message'].astype(bool)].reset_index(drop=True)

        # 2. Convert DateTime and Extract Features
        try:
            # Try parsing with AM/PM (12-hour format)
            self.df['DateTime'] = pd.to_datetime(self.df['DateTimeStr'], format='%d/%m/%y %I:%M:%S%p', errors='coerce')
        except ValueError:
            # Fallback to 24-hour format if 12-hour fails (this is usually the more robust line)
            self.df['DateTime'] = pd.to_datetime(self.df['DateTimeStr'], format='%d/%m/%y %H:%M:%S', errors='coerce')
        
        self.df.dropna(subset=['DateTime'], inplace=True)

        # Extract required temporal components
        self.df['day'] = self.df['DateTime'].dt.day
        self.df['month'] = self.df['DateTime'].dt.month_name()
        self.df['year'] = self.df['DateTime'].dt.year
        self.df['hour'] = self.df['DateTime'].dt.hour
        self.df['minute'] = self.df['DateTime'].dt.minute
        self.df['day_of_week'] = self.df['DateTime'].dt.day_name() # <-- Column is created here!

        # 3. Rename columns and Filter out system messages
        self.df.rename(columns={'Sender': 'sender', 'Message': 'message'}, inplace=True)

        system_messages = [
            'Messages and calls are end-to-end encrypted.', 'image omitted', 
            'document omitted', 'sticker omitted', 'This message was deleted.', 
            'GIF omitted', 'audio omitted', 'video omitted', 'You joined using this group\'s invite link',
            'changed the group name', 'changed this group\'s icon', 'changed the group settings',
            'created this group', 'left', 'added', 'removed', 'joined'
        ]
        
        # Filter out system messages based on content
        self.df = self.df[~self.df['message'].str.contains('|'.join(system_messages), case=False, na=False)]

        # --- KEY FIX HERE: Include 'day_of_week' in the returned DataFrame ---
        return self.df[['day', 'month', 'year', 'hour', 'minute', 'day_of_week', 'sender', 'message', 'DateTime']]


# --- 2. Analysis Engine Logic (Consolidated) ---

class WhatsAppAnalyzer:
    def __init__(self, df: pd.DataFrame, user: str = 'Overall'):
        self.df_base = df
        self.user = user
        self.df = df if user == 'Overall' else df[df['sender'] == user]
        self.text = " ".join(self.df['message'].astype(str).tolist())
        
    def get_base_metrics(self) -> Dict[str, int]:
        total_messages = len(self.df)
        total_words = sum(self.df['message'].str.split().apply(len))
        
        links = self.df['message'].apply(lambda x: len(URL_PATTERN.findall(x)))
        total_links = links.sum()
        
        return {
            'Total Messages': total_messages,
            'Total Words': total_words,
            'Links Shared': total_links
        }

    def get_active_senders_and_contribution(self) -> pd.DataFrame:
        if self.user != 'Overall':
            return pd.DataFrame()
        
        sender_counts = self.df['sender'].value_counts()
        total_messages = sender_counts.sum()
        
        df_active = sender_counts.head(5).to_frame(name='Message Count')
        df_active['Contribution (%)'] = (df_active['Message Count'] / total_messages) * 100
        df_active['Sender'] = df_active.index
        df_active.reset_index(drop=True, inplace=True)
        return df_active

    def get_top_emojis(self, top_n: int = 5) -> pd.DataFrame:
        emojis_list = [c for c in self.text if c in emoji.EMOJI_DATA]
        emoji_counts = Counter(emojis_list).most_common(top_n)
        
        df_emojis = pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])
        return df_emojis

    def get_word_frequency(self, top_n: int = 10) -> pd.DataFrame:
        words = re.findall(r'\b\w+\b', self.text.lower())
        filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 1]
        word_counts = Counter(filtered_words).most_common(top_n)
        
        df_words = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        return df_words

    def get_timeline_data(self) -> Dict[str, pd.DataFrame]:
        
        # 1. Hourly Activity
        df_hourly = self.df['hour'].value_counts().sort_index().to_frame(name='Message Count')
        df_hourly['Hour'] = df_hourly.index
        
        # 2. Day of Week Activity
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Accessing 'day_of_week' now guaranteed to exist due to fix in processor
        df_day_of_week = self.df['day_of_week'].astype('category').cat.set_categories(day_order, ordered=True).value_counts().sort_index().fillna(0).to_frame(name='Message Count')
        df_day_of_week['Day'] = df_day_of_week.index
        
        # 3. Monthly Activity
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        df_monthly = self.df['month'].astype('category').cat.set_categories(month_order, ordered=True).value_counts().sort_index().fillna(0).to_frame(name='Message Count')
        df_monthly['Month'] = df_monthly.index

        return {
            'Hourly': df_hourly.reset_index(drop=True),
            'DayOfWeek': df_day_of_week.reset_index(drop=True),
            'Monthly': df_monthly.reset_index(drop=True)
        }

    def get_top_lengthy_messages(self, top_n: int = 3) -> pd.DataFrame:
        df_lengthy = self.df.copy()
        df_lengthy['Word_Count'] = df_lengthy['message'].str.split().apply(len)
        df_lengthy = df_lengthy.sort_values(by='Word_Count', ascending=False).head(top_n)
        
        if self.user == 'Overall':
            return df_lengthy[['sender', 'DateTime', 'message', 'Word_Count']].rename(columns={'DateTime': 'Time Stamp'})
        else:
            return df_lengthy[['DateTime', 'message', 'Word_Count']].rename(columns={'DateTime': 'Time Stamp'})


# --- 3. Streamlit Application Logic (Consolidated) ---

@st.cache_data(show_spinner="Processing chat data...")
def load_and_process_data(uploaded_file):
    """Caches file processing for efficiency."""
    if uploaded_file is None:
        return None
    
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    processor = WhatsAppDataProcessor(stringio)
    processor.parse_chat()
    df_processed = processor.process_dataframe()
    
    return df_processed if not df_processed.empty else None

def display_analysis_results(df: pd.DataFrame, selected_user: str):
    """Displays all the analysis sections."""
    
    analyzer = WhatsAppAnalyzer(df, selected_user)
    
    # 1. Core Metrics
    st.subheader("📊 Core Metrics: Volume, Words, and Links")
    metrics = analyzer.get_base_metrics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Messages", value=metrics['Total Messages'])
    with col2:
        st.metric(label="Total Words", value=metrics['Total Words'])
    with col3:
        st.metric(label="Links Shared", value=metrics['Links Shared'])
    st.divider()
    
    # 2. Most Active Sender (Overall Only)
    if selected_user == 'Overall':
        st.subheader("👥 Most Active Senders (Top 5)")
        df_active = analyzer.get_active_senders_and_contribution()
        
        if not df_active.empty:
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig = px.bar(
                    df_active,
                    x='Sender',
                    y='Message Count',
                    color='Contribution (%)',
                    text='Contribution (%)',
                    title="Message Count by Top Senders",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.markdown("##### Contribution Table")
                df_active['Contribution (%)'] = df_active['Contribution (%)'].round(1).astype(str) + '%'
                st.dataframe(df_active, use_container_width=True, hide_index=True)
        st.divider()
        
    # 3. Timelines (Peak Hours, Day, Month)
    st.subheader("🕒 Message Activity Timelines (Peak Activity)")
    timeline_data = analyzer.get_timeline_data()
    
    st.markdown("##### Peak Active Hours (24-Hour Clock)")
    fig_hour = px.line(
        timeline_data['Hourly'], 
        x='Hour', 
        y='Message Count', 
        markers=True, 
        title=f"Hourly Activity for {selected_user}",
        template="plotly_white",
        color_discrete_sequence=['#1f77b4']
    )
    fig_hour.update_xaxes(tick0=0, dtick=1)
    st.plotly_chart(fig_hour, use_container_width=True)

    col_day, col_month = st.columns(2)
    
    with col_day:
        st.markdown("##### Activity by Day of Week")
        fig_day = px.bar(
            timeline_data['DayOfWeek'],
            x='Day',
            y='Message Count',
            title=f"Activity by Day of Week for {selected_user}",
            template="plotly_white",
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_day, use_container_width=True)

    with col_month:
        st.markdown("##### Activity by Month")
        df_monthly = timeline_data['Monthly']
        df_monthly = df_monthly[df_monthly['Message Count'] > 0]
        fig_month = px.bar(
            df_monthly,
            x='Month',
            y='Message Count',
            title=f"Activity by Month for {selected_user}",
            template="plotly_white",
            color_discrete_sequence=['#d62728']
        )
        st.plotly_chart(fig_month, use_container_width=True)
    st.divider()

    # 4. Word and Emoji Analysis
    st.subheader("🔎 Word and Emoji Analysis")
    
    col_topwords, col_emojis = st.columns([1, 2])
    
    with col_topwords:
        st.markdown("##### Top 10 Common Words")
        df_words = analyzer.get_word_frequency(top_n=10)
        st.dataframe(df_words, use_container_width=True, hide_index=True)
    
    with col_emojis:
        st.markdown("##### Top 5 Emojis Used (Count)")
        df_emojis = analyzer.get_top_emojis(top_n=5)
        
        if df_emojis.empty:
            st.info("No emojis found in the messages.")
        else:
            fig_emoji = px.pie(
                df_emojis, 
                values='Count', 
                names='Emoji', 
                title=f"Emoji Distribution for {selected_user}",
                template="plotly_white"
            )
            fig_emoji.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_emoji, use_container_width=True)
            
    st.markdown("##### Word Cloud of Common Terms")
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=CUSTOM_STOPWORDS, 
            min_font_size=5,
            colormap='viridis'
        ).generate(analyzer.text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception:
        st.info("Could not generate WordCloud. Insufficient cleaned text data.")

    st.divider()
    
    # 5. Longest Messages
    st.subheader(f"💬 Top 3 Lengthiest Messages (by word count)")
    df_lengthy = analyzer.get_top_lengthy_messages(top_n=3)

    if df_lengthy.empty:
        st.info("No messages found to determine the longest ones.")
        return

    for index, row in df_lengthy.iterrows():
        sender_info = f"**{row['sender']}**" if 'sender' in row else selected_user
        snippet = row['message'][:500] + '...' if len(row['message']) > 500 else row['message']
        
        with st.container(border=True):
            st.markdown(f"**Word Count:** `{row['Word_Count']}`")
            if selected_user == 'Overall':
                 st.markdown(f"**Sender:** {sender_info} | **Time:** {row['Time Stamp'].strftime('%Y-%m-%d %H:%M')}")
            else:
                 st.markdown(f"**Time:** {row['Time Stamp'].strftime('%Y-%m-%d %H:%M')}")
                 
            st.markdown(f"***Message Snippet:***")
            st.code(snippet, language='text')

# --- Main Application Runner ---

def main():
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💬 WhatsApp Chat Analysis Dashboard")
    st.markdown("Upload your exported WhatsApp chat text file (`_chat.txt`) to explore the data.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload **_chat.txt** File", 
        type=['txt'], 
        help="Please ensure your chat is exported without media for the best parsing results."
    )

    if uploaded_file is None:
        st.info("Awaiting file upload...")
        return

    # --- Data Loading and Caching ---
    df = load_and_process_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("Could not process the chat file or no clean messages were found. Please verify the file format.")
        return

    # --- Sidebar for User Selection ---
    sender_list = ['Overall'] + sorted(df['sender'].unique().tolist())
    
    st.sidebar.header("User Selection")
    selected_user = st.sidebar.selectbox(
        "Choose User for Analysis", 
        sender_list
    )
    
    st.sidebar.markdown(f"### Current View: **{selected_user}**")

    # --- Analysis Execution ---
    display_analysis_results(df, selected_user)
    
    st.markdown("---")
    st.markdown("### Processed Data Sample (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

if __name__ == "__main__":
    main()
