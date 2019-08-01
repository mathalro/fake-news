import re

class RemoveNoise:

    def __init__(self):
        pass

    # Remove noise characters
    def execute(self, dataframe, execute):

        if (execute == True):
            print("Remove bad characters")
            for index, row in dataframe.iterrows():
                row['text'] = re.sub('[0-9\[\](){}=.,:;+?/!\*<>_\-§%\$\'\"“”]', '', row['text'])
                row['text'] = re.sub('(www|http)\S+', '', row['text'])

        return dataframe