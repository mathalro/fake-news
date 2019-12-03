class LowerCase:

    def _init__(self):
        pass

    # Lowercasing
    def execute(self, dataframe, execute):

        if (execute == True):
            print("Lower case all text")

            for index, row in dataframe.iterrows():
                row['text'] = row['text'].lower()
                if (row['label'] != None):
                    row['label'] = row['label'].lower()

        return dataframe