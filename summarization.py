# -*- coding: utf-8 -*-
"""Summarization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ltRxG88pjapGP9W0rJizeDEfs9eF1fNO
"""

!pip install lxml

!pip install nltk

!pip install beautifulsoup4

import bs4 as bs
import urllib.request
import re

userLink = input("Which Wikipedia article would you want me to summarize: ")
raw_data = urllib.request.urlopen(userLink)
document = raw_data.read()

parsed_document = bs.BeautifulSoup(document,'lxml')

particle_paras = parsed_document.find_all('p')

scrapped_data = ""

for para in particle_paras:
    scrapped_data += para.text

print(scrapped_data[:100])

scrapped_data = re.sub(r'\[[0-9]*\]', ' ',  scrapped_data)
scrapped_data = re.sub(r'\s+', ' ',  scrapped_data)

import nltk
nltk.download('punkt')

all_sentences = nltk.sent_tokenize(scrapped_data)

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

word_freq = {}
for word in nltk.word_tokenize(scrapped_data):
    if word not in stopwords:
        if word not in word_freq.keys():
            word_freq[word] = 1
        else:
            word_freq[word] += 1

sentence_scores = {}
for sentence in all_sentences:
    for token in nltk.word_tokenize(sentence.lower()):
        if token in word_freq.keys():
            if len(sentence.split(' ')) <25:
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = word_freq[token]
                else:
                    sentence_scores[sentence] += word_freq[token]

import heapq
selected_sentences= heapq.nlargest(50, sentence_scores, key=sentence_scores.get)

text_summary = ' '.join(selected_sentences)
print(text_summary)