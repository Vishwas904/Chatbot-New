from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
import json
import pickle
import numpy as np
import tensorflow as tf
from spellchecker import SpellChecker

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents from the JSON file
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

# Create a spell checker instance
spell = SpellChecker()

# Load sentiment analysis model and tokenizer
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

def analyze_sentiment(message):
    inputs = sentiment_tokenizer(message, return_tensors="pt")
    outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    sentiment_label = torch.argmax(probabilities, dim=1).item()
    return sentiment_label

def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    # Use spell checker to correct misspelled words
    corrected_words = [spell.correction(word) for word in sentence_words]

    return corrected_words

def bag_of_words(sentence):
    sentence_words = cleanup_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent.get('tags') == tag:
                return intent['response']
    except IndexError:
        print("I'm sorry, I don't understand your question.")
    # return "I'm sorry, I don't understand your question."

print("Chatbot is working")

while True:
    message = input("You: ")
    if message.lower() in ["quit", "exit"]:
        break

    # Correct misspelled words in the user's query
    corrected_message = ' '.join(cleanup_sentence(message.lower()))
    
    # Perform sentiment analysis on the user input
    sentiment_label = analyze_sentiment(corrected_message)
    
    if sentiment_label == 1:  # Positive sentiment
        # Classify intents and get a response
        classified_intents = predict_class(corrected_message)
        response = get_response(classified_intents, intents)
        print("Bot:", response)
    else:
        print("I do not have an answer to this question. Do you want to raise a query regarding this?")
