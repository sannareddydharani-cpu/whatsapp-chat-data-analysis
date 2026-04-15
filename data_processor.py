import pandas as pd
import re
from io import StringIO
from typing import List, Dict, Any

# Regular expression to parse the chat log line.
# It handles the common WhatsApp export format.
WA_MESSAGE_START_PATTERN = re.compile(
    r'^\[(\d{2}/\d{2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2})\s*(\u202f|)(AM|PM|am|pm|)\s*\]\s*([^:]+):\s*(.*)',
    re.MULTILINE | re.IGNORECASE
)

class WhatsAppDataProcessor:
    """
    Handles reading the uploaded file and transforming the raw text into a clean DataFrame
    ready for analysis.
    """
    def __init__(self, uploaded_file: StringIO):
        """Initializes with the raw file content."""
        self.file_content = uploaded_file.getvalue()
        self.df = pd.DataFrame()

    def parse_chat(self) -> None:
        """Parses the raw text content line by line, handling multiline messages."""
        data: List[Dict[str, Any]] = []
        current_message: Dict[str, Any] = {}
        lines = self.file_content.splitlines()

        for line in lines:
            match = WA_MESSAGE_START_PATTERN.match(line)

            if match:
                # Store the previous message before starting a new one
                if current_message:
                    data.append(current_message)

                # Extract date, time, sender, and message start
                date_part = match.group(1)
                time_ampm_part = f"{match.group(3)}{match.group(4).strip()}" if match.group(4) else ""
                time_part = f"{match.group(2)}{time_ampm_part}"

                current_message = {
                    'DateTimeStr': f"{date_part} {time_part}",
                    'Sender': match.group(5).strip(),
                    'Message': match.group(6).strip()
                }
            else:
                # Continuation of the previous message
                if current_message:
                    current_message['Message'] += '\n' + line.strip()
                # Ignoring non-message lines (e.g., chat file headers)
                continue

        if current_message:
            data.append(current_message)

        self.df = pd.DataFrame(data)

    def process_dataframe(self) -> pd.DataFrame:
        """
        Cleans the DataFrame, extracts temporal features, and filters out system messages.
        """
        if self.df.empty:
            return self.df

        # 1. Clean and Prepare
        self.df['Message'] = self.df['Message'].str.replace(r'^\u200e|\u200f', '', regex=True).str.strip()
        self.df.dropna(subset=['Message'], inplace=True)
        self.df = self.df[self.df['Message'].astype(bool)].reset_index(drop=True)

        # 2. Convert DateTime and Extract Features
        try:
            # Try common formats that handle 12h clock with AM/PM
            self.df['DateTime'] = pd.to_datetime(self.df['DateTimeStr'], format='%d/%m/%y %I:%M:%S%p', errors='coerce')
        except ValueError:
            # Fallback for 24h clock format or other common variations
            self.df['DateTime'] = pd.to_datetime(self.df['DateTimeStr'], format='%d/%m/%y %H:%M:%S', errors='coerce')
        
        self.df.dropna(subset=['DateTime'], inplace=True)

        # Extract required temporal components
        self.df['day'] = self.df['DateTime'].dt.day
        self.df['month'] = self.df['DateTime'].dt.month_name()
        self.df['year'] = self.df['DateTime'].dt.year
        self.df['hour'] = self.df['DateTime'].dt.hour
        self.df['minute'] = self.df['DateTime'].dt.minute
        self.df['day_of_week'] = self.df['DateTime'].dt.day_name()

        # 3. Rename columns to meet the exact requirement
        self.df.rename(columns={'Sender': 'sender', 'Message': 'message'}, inplace=True)

        # 4. Filter out system messages and media omissions
        system_messages = [
            'Messages and calls are end-to-end encrypted.', 'image omitted', 
            'document omitted', 'sticker omitted', 'This message was deleted.', 
            'GIF omitted', 'audio omitted', 'video omitted', 'You joined using this group\'s invite link'
        ]
        self.df = self.df[~self.df['message'].str.contains('|'.join(system_messages), case=False, na=False)]

        return self.df[['day', 'month', 'year', 'hour', 'minute', 'sender', 'message', 'DateTime']]
