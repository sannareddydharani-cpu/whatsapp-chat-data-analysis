import pandas as pd
import re
from collections import Counter
import emoji
from wordcloud import STOPWORDS
from typing import Dict, Any, List

# --- Constants for Analysis ---

# Regex to detect links in messages
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# A set of common English and Hindi/Hinglish stop words
CUSTOM_STOPWORDS = set(STOPWORDS)
CUSTOM_STOPWORDS.update([
    'media', 'omitted', 'message', 'deleted', 'pm', 'am', 'hai', 'kya', 'bhai', 'bhi', 'hum', 'aur', 
    'sir', 'guys', 'apni', 'apne', 'sath', 'jo', 'woh', 'yeh', 'kon', 'kar', 'karke', 'kiya', 'bhi', 
    'liye', 'ko', 'per', 'pe', 'hai', 'nahi', 'par', 'from', 'to', 'for', 'the', 'is', 'it', 'and', 
    'in', 'of', 'i', 'you', 'me', 'will', 'was', 'we', 'are', 'was', 'were', 'have', 'had', 'has', 
    'do', 'did', 'does', 'can', 'could', 'would', 'should', 'get', 'like', 'with', 'about', 'just',
    'one', 'only', 'may', 'must', 'ask', 'them', 'this', 'that', 'plz', 'please', 'aap', 'sir', 
    'mam', 'b', 'c', 'msg', 'whatsapp', 'messages', 'call' # Added more common chat words
])

class WhatsAppAnalyzer:
    """
    Performs various statistical and linguistic analyses on the message DataFrame,
    supporting both overall and individual user scopes.
    """
    def __init__(self, df: pd.DataFrame, user: str = 'Overall'):
        """Filters the DataFrame based on the selected user."""
        self.df_base = df
        self.user = user
        self.df = df if user == 'Overall' else df[df['sender'] == user]
        self.text = " ".join(self.df['message'].astype(str).tolist())
        
    def get_base_metrics(self) -> Dict[str, int]:
        """Calculates total messages, total words, and links shared."""
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
        """(Overall only) Returns top 5 active senders and their % contribution."""
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
        """Extracts and counts the top N most used emojis."""
        emojis_list = [c for c in self.text if c in emoji.EMOJI_DATA]
        emoji_counts = Counter(emojis_list).most_common(top_n)
        
        df_emojis = pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])
        return df_emojis

    def get_word_frequency(self, top_n: int = 10) -> pd.DataFrame:
        """Calculates and returns the top N most common non-stop words."""
        # Finds all contiguous word characters and converts to lowercase
        words = re.findall(r'\b\w+\b', self.text.lower())
        filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 1]
        word_counts = Counter(filtered_words).most_common(top_n)
        
        df_words = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        return df_words

    def get_timeline_data(self) -> Dict[str, pd.DataFrame]:
        """Calculates message activity across hours, days of week, and months."""
        
        # 1. Hourly Activity
        df_hourly = self.df['hour'].value_counts().sort_index().to_frame(name='Message Count')
        df_hourly['Hour'] = df_hourly.index
        
        # 2. Day of Week Activity
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Use a categorical type for correct sorting, then count
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
        """Returns the top N longest messages based on word count."""
        df_lengthy = self.df.copy()
        df_lengthy['Word_Count'] = df_lengthy['message'].str.split().apply(len)
        df_lengthy = df_lengthy.sort_values(by='Word_Count', ascending=False).head(top_n)
        
        # Prepare columns for display
        if self.user == 'Overall':
            return df_lengthy[['sender', 'DateTime', 'message', 'Word_Count']].rename(columns={'DateTime': 'Time Stamp'})
        else:
            return df_lengthy[['DateTime', 'message', 'Word_Count']].rename(columns={'DateTime': 'Time Stamp'})
