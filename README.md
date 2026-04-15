# WhatsApp Chat Analyzer: Interactive Streamlit Dashboard 💬📊
This is a comprehensive, interactive Streamlit web application for analyzing exported WhatsApp chat data (.txt files). It transforms unstructured text data into actionable metrics and insightful visualizations, allowing users to deeply understand chat behavior, activity trends, and language usage for both the entire group and individual members.

## ✨ Key Features
* Data Processing & Cleaning: Robust parsing of WhatsApp's chat format (including date, time, and multi-line messages) into a structured Pandas DataFrame.

* User/Overall Toggle: A sidebar dropdown to switch instantly between Overall Group Analysis and Individual User Analysis.

* Core Metrics: Calculation of total messages, word count, and external links shared.

* Activity Timelines: Interactive Plotly graphs showing peak activity by Hour, Day of the Week, and Month.

* Sender Analysis (Overall): Bar chart and percentage contribution of the Top 5 Most Active Senders.

## Linguistic Analysis:

* Word Clouds: Visual representation of the most frequently used words.

* Top Words: List of the top 10 most common non-stop words (English & Hinglish).

* Emoji Analysis: Pie/Bar charts showing the count and distribution of the Top 5 Emojis used.

* Longest Messages: Display of the Top 3 longest messages by word count.

## 🛠️ Technology Stack
* Component	Technology	Role
* Web App Framework	Streamlit	Creates the interactive web interface and handles the application flow.
* Data Processing	Pandas	Core library for data cleaning, DataFrame manipulation, and feature extraction.
* Parsing	re (Regex)	Used for robust extraction of date, time, and sender from the raw text lines.
* Visualization	Plotly Express & Matplotlib	Generates all interactive (Plotly) and static (WordCloud via Matplotlib) charts.
* Text Utilities	wordcloud & emoji	Libraries used for natural language and character processing.

## Export to Sheets
# 🚀 Setup and Installation
Follow these steps to clone the repository and run the application locally.

Clone the Repository:

Bash

git clone https://your-github-username/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
Create and Activate Virtual Environment:

Bash

python -m venv myenv
* On Windows:


.\myenv\Scripts\activate
*  On macOS/Linux:

source myenv/bin/activate
Install Dependencies:
Make sure you install the pinned version of numpy first to avoid compatibility errors.

Bash

pip install -r requirements.txt
Run the Streamlit App:

Bash

streamlit run streamlit_app.py
Your web browser will open automatically to the application URL (http://localhost:8501).

📝 How to Use
Export Your Chat: Open any WhatsApp chat, go to More Options -> More -> Export chat -> WITHOUT MEDIA. This generates the necessary .txt file.

Upload: Click the "Upload _chat.txt File" button on the Streamlit page.

Analyze: Use the User Selection dropdown in the sidebar to choose between "Overall" analysis or the statistics for any individual sender.