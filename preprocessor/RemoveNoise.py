import re
import unicodedata

class RemoveNoise:

    def __init__(self):
        pass

    def strip_accents(self, text):
        text = unicodedata.normalize('NFD', text)\
            .encode('ascii', 'ignore')\
            .decode("utf-8")

        return str(text)

    # Remove noise characters
    def execute(self, dataframe, execute, removeDiacritics = False):

        if (execute == True):
            print("Remove bad characters")
            for index, row in dataframe.iterrows():
                row['text'] = re.sub('[…0-9\[\](){}=.,:;+?/!\*<>_\-§%\$\'\"“”]', '', row['text'])
                row['text'] = re.sub('(www|http)\S+', '', row['text'])
                if removeDiacritics:
                    row['text'] = self.strip_accents(row['text'])

        return dataframe

    # TODO Create a parametrized regesx execute