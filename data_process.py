# data_process.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords
import os
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to load positive and negative words from the Master Dictionary folder
def load_master_dictionary():
    master_dict_folder = "MasterDictionary"
    
    positive_words = set()
    negative_words = set()
    
    # Load positive words
    with open(os.path.join(master_dict_folder, "positive-words.txt"), 'r', encoding='utf-8', errors='ignore') as f:
        positive_words.update([word.strip().lower() for word in f if word.strip()])
    
    # Load negative words
    with open(os.path.join(master_dict_folder, "negative-words.txt"), 'r', encoding='utf-8', errors='ignore') as f:
        negative_words.update([word.strip().lower() for word in f if word.strip()])
    
    return positive_words, negative_words

# Function to generate a summary using Hugging Face transformers
def generate_summary(text, model, tokenizer, device):
    if not text or not isinstance(text, str) or text.strip() == '':
        return 'No content available to summarize.'
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs, 
        max_length=500, 
        min_length=200, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to clean text and tokenize
def clean_text(text, stop_words):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return tokens

# Sentiment scoring functions
def positive_score(tokens, positive_words):
    return sum(1 for word in tokens if word in positive_words)

def negative_score(tokens, negative_words):
    return sum(1 for word in tokens if word in negative_words)

def polarity_score(positive, negative):
    return (positive - negative) / ((positive + negative) + 0.000001)

def subjectivity_score(positive, negative, total_words):
    return (positive + negative) / (total_words + 0.000001)

def average_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    if len(sentences) == 0:
        return 0
    return len(words) / len(sentences)

def count_syllables(word):
    word = word.lower()
    vowels = "aeiou"
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
                prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if word.endswith("e"):
        count = max(1, count - 1)
    return count

def percentage_complex_words(tokens):
    if not tokens:
        return 0
    complex_words = [word for word in tokens if count_syllables(word) > 2]
    return len(complex_words) / len(tokens)

def fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

def process_article(article_text, stop_words, positive_words, negative_words):
    tokens = clean_text(article_text, stop_words)
    
    pos_score = positive_score(tokens, positive_words)
    neg_score = negative_score(tokens, negative_words)
    pol_score = polarity_score(pos_score, neg_score)
    subj_score = subjectivity_score(pos_score, neg_score, len(tokens))
    
    avg_sent_len = average_sentence_length(article_text)
    perc_complex_words = percentage_complex_words(tokens)
    fog_idx = fog_index(avg_sent_len, perc_complex_words)
    
    return {
        "Positive Score": pos_score,
        "Negative Score": neg_score,
        "Polarity Score": pol_score,
        "Subjectivity Score": subj_score,
        "Avg Sentence Length": avg_sent_len,
        "Percentage of Complex Words": perc_complex_words,
        "Fog Index": fog_idx,
    }
