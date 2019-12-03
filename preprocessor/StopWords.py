import nltk
from nltk.corpus import stopwords

class StopWords:
    
    # Remove stop words
    def execute(self, dataframe, execute):

        if (execute == True):
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

        return dataframe
