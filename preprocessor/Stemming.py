import nltk

class Stemming:

    # Stemming
    def execute(self, dataframe, execute):
        
        if (execute == True):
            print("Stemming words")

            nltk.download('rslp')
            stemmer = nltk.stem.RSLPStemmer()

            for index, row in dataframe.iterrows():
                text = ""
                for w in row['text'].split():
                    text += stemmer.stem(w) + " "  

                row['text'] = text

        return dataframe