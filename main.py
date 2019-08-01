# from collector.Collector import Collector 

# c = Collector()
# data = c.collect()

# print("\n\n\nFound data...\n\n\n")

# for i in data:
#     print(i.content.text)
#     print(i.label)
#     print()

import pandas as pd

from preprocessor.RemoveNoise import RemoveNoise
from preprocessor.LowerCase import LowerCase
from preprocessor.StopWords import StopWords
from preprocessor.Stemming import Stemming

dataframe = pd.read_csv('CollectorLupa', sep='\t', names=['text', 'label'])

removeNoise = RemoveNoise()
lowerCase = LowerCase()
stopWords = StopWords()
stemming = Stemming()

removeNoise.execute(dataframe, True)
lowerCase.execute(dataframe, True)
stopWords.execute(dataframe, True)
stemming.execute(dataframe, True)

print(dataframe)
