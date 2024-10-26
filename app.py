# app.py

import streamlit as st
import pandas as pd
import torch
import datetime

from data_collect import fetch_news_articles
from data_process import generate_summary, process_article, load_master_dictionary

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load positive and negative words
positive_words, negative_words = load_master_dictionary()

# Hugging Face summarization model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

model_name = 'sshleifer/distilbart-cnn-12-6'  # Smaller model for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)

# Set up the Streamlit layout
st.title("Indian Newspaper Article Collector and Analyzer")

st.sidebar.header("Filters")

# Dropdown for selecting newspaper name
newspaper_names = [
    'Hindustan Times', 'Indian Express', 'Telegraph India', 'Deccan Chronicle',
    'New Indian Express', 'Live Mint', 'Business Standard', 'Financial Express', 'DNA India',
    'The Tribune', 'The Statesman', 'Asian Age', 'Daily Pioneer', 'Free Press Journal',
    'Economic Times', 'The Hans India', 'Orissa Post', 'The Hitavada', 'The Sentinel Assam',
    'Navhind Times', 'Assam Tribune', 'Arunachal Times', 'Shillong Times', 'Sanga Express'
]
newspaper_name = st.sidebar.selectbox('Select Newspaper', newspaper_names)

# Slider to select the number of articles to fetch per newspaper
article_limit = st.sidebar.slider('Number of articles to fetch', min_value=1, max_value=10, value=5)

# Date filter
start_date = st.sidebar.date_input("Start Date", value=datetime.date.today())
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

# Option to include articles without a published date
include_no_date = st.sidebar.checkbox("Include articles without a published date", value=True)

# Button to trigger news fetching
if st.button("Fetch News Articles"):
    st.write(f"Fetching articles for {newspaper_name}...")
    
    articles_df = fetch_news_articles(newspaper_name, article_limit)
    
    if articles_df.empty:
        st.write(f"No articles found for {newspaper_name}. Please try selecting another newspaper.")
    else:
        # Filter by date range
        articles_df['Published Date'] = pd.to_datetime(articles_df['Published Date'], errors='coerce')
        
        if include_no_date:
            # Include articles without a published date
            filtered_articles = articles_df[
                ((articles_df['Published Date'] >= pd.to_datetime(start_date)) & 
                 (articles_df['Published Date'] <= pd.to_datetime(end_date))) |
                (articles_df['Published Date'].isna())
            ]
        else:
            # Exclude articles without a published date
            filtered_articles = articles_df[
                (articles_df['Published Date'] >= pd.to_datetime(start_date)) & 
                (articles_df['Published Date'] <= pd.to_datetime(end_date))
            ]
        
        if filtered_articles.empty:
            st.write("No articles found for the selected date range.")
        else:
            st.write(f"Displaying {len(filtered_articles)} articles:")
            for index, row in filtered_articles.iterrows():
                with st.expander(f"{row['Headline']}"):
                    st.subheader("Summary")
                    summary = generate_summary(row['Content'], model, tokenizer, device)
                    st.write(summary)
                    
                    st.subheader("Full Article")
                    st.write(row['Content'])
                    
                    # Perform sentiment analysis and text metrics
                    analysis = process_article(row['Content'], stop_words, positive_words, negative_words)
                    st.subheader("Text Analysis Metrics")
                    st.write(f"Positive Score: {analysis['Positive Score']}")
                    st.write(f"Negative Score: {analysis['Negative Score']}")
                    st.write(f"Polarity Score: {analysis['Polarity Score']}")
                    st.write(f"Subjectivity Score: {analysis['Subjectivity Score']}")
                    st.write(f"Avg Sentence Length: {analysis['Avg Sentence Length']}")
                    st.write(f"Percentage of Complex Words: {analysis['Percentage of Complex Words']}")
                    st.write(f"Fog Index: {analysis['Fog Index']}")
