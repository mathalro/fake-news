# from collector.Collector import Collector 

# c = Collector()
# data = c.collect()

# print("\n\n\nFound data...\n\n\n")

# for i in data:
#     print(i.content.text)
#     print(i.label)
#     print()

import pandas as pd
import re

import nltk
from nltk.corpus import stopwords 

dataframe = pd.read_csv('CollectorLupa', sep='\t', names=['text', 'label'])

# Remove noise characters
print("Remove bad characters")
for index, row in dataframe.iterrows():
    row['text'] = re.sub('[0-9\[\](){}=.,:;+?/!\*<>_\-§%\$\'\"“”]', '', row['text'])
    row['text'] = re.sub('(www|http)\S+', '', row['text'])


# Lowercasing
print("Lower case all text")

for index, row in dataframe.iterrows():
    row['text'] = row['text'].lower()
    row['label'] = row['label'].lower()


# Remove stop words
print("Remove stop words")

nltk.download('stopwords')
nltk_stopwords = list(stopwords.words('portuguese'))

for index, row in dataframe.iterrows():
    text = ""
    for w in row['text'].split():
        is_stop = False
        for stop in nltk_stopwords:
            if w in nltk_stopwords:
                is_stop = True
                
        if not is_stop:
            text += w + " "

    row['text'] = text


# Stemming
print("Stemming words")

nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

for index, row in dataframe.iterrows():
    text = ""
    for w in row['text'].split():
        text += stemmer.stem(w) + " "  

    row['text'] = text

print(dataframe)
