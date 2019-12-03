# from collector.Collector import Collector 

# c = Collector()
# data = c.collect()

# print("\n\n\nFound data...\n\n\n")

# for i in data:
#     print(i.content.text)
#     print(i.label)
#     print()

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics

from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD
from sklearn import model_selection, naive_bayes, svm
from keras import callbacks

from preprocessor.RemoveNoise import RemoveNoise
from preprocessor.LowerCase import LowerCase
from preprocessor.StopWords import StopWords
from preprocessor.Stemming import Stemming


def plot_history(history, title):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure()
    plt.title(title)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Acurácia de treinamento')
    plt.plot(x, val_acc, 'r', label='Acurácia de validação')
    plt.title('Acurácia')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Erro de treinamento')
    plt.plot(x, val_loss, 'r', label='Erro de validação')
    plt.title('Erro')
    plt.legend()
    plt.savefig(title+".png")

def print_metrics(model, X, y, type = "rna"):
    y_pred = []
    y_true = []
    pred = model.predict(X)

    true_fake = 0
    false_fake = 0
    true_real = 0
    false_real = 0

    for i in range(len(pred)):
        val = 0

        if type != "rna":
            val = pred[i]
        else:
            val = pred[i][0]

        y_pred.append(int(round(val)))
        y_true.append(y[i])

        if (y_true[i] == 1 and y_pred[i] == 1):
            true_fake += 1

        if (y_true[i] == 0 and y_pred[i] == 1):
            false_fake += 1

        if (y_true[i] == 0 and y_pred[i] == 0):
            true_real += 1
        
        if (y_true[i] == 1 and y_pred[i] == 0):
            false_real += 1

    # print(true_fake)
    # print(false_fake)
    # print(true_real)
    # print(false_real)

    # print(y_true)
    # print(y_pred)

    print("Confusion matrix:")
    print(pd.DataFrame(metrics.confusion_matrix(y_true, y_pred, labels=[1, 0]), index=['true:fake', 'true:real'], columns=['pred:fake', 'pred:real']))
    print("\nAccuracy:", metrics.accuracy_score(y_true, y_pred))
    print("Precision:", metrics.precision_score(y_true, y_pred))
    print("Recall:", metrics.recall_score(y_true, y_pred))
    print("F1 score:", metrics.f1_score(y_true, y_pred))
    print("\n==================================================================================\n\n")

def train_test_ds(x, y, test_size=0.20):
    #sentences = df['text'].values
    #y = df['label'].values

    sentences = x
    y = y

    #X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=test_size, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=test_size, stratify=y)

    return [X_train, X_test, y_train, y_test]

def ann_classifier(X_train, X_test, y_train, y_test, neuronRate = 1, epochs = 3, layers_qtd = 1, batch = 64):
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    input_dim = X_train.shape[1]
    L = input_dim
    L = int(L*neuronRate)

    model = Sequential()
    
    #kernel_regularizer=regularizers.l2(0.01)

    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu'))
    for i in range(0, layers_qtd):
        model.add(layers.Dense(L, input_dim=L, kernel_regularizer=regularizers.l2(0.01), activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs, verbose=True,
                        validation_data=(X_test, y_test), batch_size = 64)

    plot_history(history, "Rede neural")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("\n\nRNA RESULTS\n")
    print_metrics(model, X_test, y_test)

    return accuracy


def cnn_classifier(X_train, X_test, y_train, y_test, n_filters = 50, epochs = 3, dim = 100):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    maxSampleSize = 0
    for sample in X_train:
        if len(sample) > maxSampleSize:
            maxSampleSize = len(sample)

    print(maxSampleSize)

    vocab_size = len(tokenizer.word_index) + 1
    maxlen = maxSampleSize
    embedding_dim = dim

    X_train = pad_sequences(X_train, padding='post', truncating='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', truncating='post', maxlen=maxlen)

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=dim, input_length=maxlen))
    model.add(layers.Conv1D(n_filters, kernel_size=5, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPool1D())
    model.add(layers.Flatten())
    
    model.add(layers.Dense(1000, input_dim=1000, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    es_callback = callbacks.EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(X_train, y_train, epochs=epochs, verbose=True, validation_data=(X_test, y_test), batch_size=64, callbacks=[es_callback])
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

    print("\n\nCNN RESULTS\n")
    print_metrics(model, X_test, y_test)
    #plot_history(history, "Common Neural Network")

    return accuracy

def svm_classifier(X_train, X_test, y_train, y_test, c, gamma):
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    input_dim = X_train.shape[1]
    L = input_dim
 
    SVM = svm.SVC(C=c, kernel='rbf', gamma=gamma, random_state=0)
    SVM.fit(X_train, y_train)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(X_test)
    # Use accuracy_score function to get the accuracy
    #print("SVM Accuracy Score -> ", metrics.accuracy_score(predictions_SVM, y_test)*100)
    acc = metrics.accuracy_score(predictions_SVM, y_test)*100
    if acc > 70:
        print_metrics(SVM, X_test, y_test, "svm")


def svm_classifier(X_train, X_test, y_train, y_test):
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    input_dim = X_train.shape[1]
    L = input_dim
 
    clf = svm.SVC(C=c, kernel='rbf', gamma=gamma, random_state=0)
    clf.fit(X_train, y_train)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(X_test)
    # Use accuracy_score function to get the accuracy
    #print("SVM Accuracy Score -> ", metrics.accuracy_score(predictions_SVM, y_test)*100)
    acc = metrics.accuracy_score(predictions_SVM, y_test)*100
    if acc > 70:
        print_metrics(SVM, X_test, y_test, "svm")

lupa = pd.read_csv('./data/CollectorLupa', sep='\t', names=['text', 'label'])
aosfatos = pd.read_csv('./data/CollectorAosfatos', sep='\t', names=['text', 'label'])

dataframe = lupa.append(aosfatos)

removeNoise = RemoveNoise()
lowerCase = LowerCase()
stopWords = StopWords()
stemming = Stemming()
removeNoise.execute(dataframe, True, True)

lowerCase.execute(dataframe, True)
stopWords.execute(dataframe, True)

true = 0
false = 0

todelete = []

x = []
y = []

df = pd.DataFrame(columns=['text', 'label'])

for index, row in dataframe.iterrows():
    if row['label'].lower() == "verdadeiro":
        true += 1
        row['label'] = 0
        x.append(row['text'])
        y.append(row['label'])
    elif row['label'].lower() == "falso":
        row['label'] = 1
        false += 1
        x.append(row['text'])
        y.append(row['label'])
    else:
        todelete.append(index)

data = train_test_ds(x, y, test_size=0.2)

test_true = 0
test_false = 0
for i in range (len(data[3])):
    val = data[3][i]
    if (val == 0):
        test_true += 1
    else:
        test_false += 1

train_true = 0
train_false = 0
for i in range (len(data[2])):
    val = data[2][i]
    if (val == 0):
        train_true += 1
    else:
        train_false += 1
       
print("TRAIN FALSE: ", train_false)
print("TRAIN TRUE", train_true)

print("TEST FALSE: ", test_false)
print("TEST TRUE", test_true)


# for i in range(10, 21):
#     gamma = i * 0.01
#     for j in range(1, 11):
#         c = j * 0.1
#         print(gamma, c)
#         res = svm_classifier(data[0], data[1], data[2], data[3], c, gamma)
#         #print("%.2f" % round(res,2) + ";", end = '')
#     #print()


#ann_classifier(data[0], data[1], data[2], data[3], 0.5, 10, 1, 64)

resultMatrix = []
for i in range(4, 6):
    layers_qtd = i
    lineResult = []
    for j in range(16, 129, 16):
        batch = j
        sum = 0
        qtd = 3
        print("Test", layers_qtd, batch)
        for q in range(0, qtd):
            sum += ann_classifier(data[0], data[1], data[2], data[3], 0.5, 3, layers_qtd, batch)
        lineResult.append(sum/qtd)
    resultMatrix.append(lineResult)


# resultMatrix = []
# for i in range(170, 171, 20):
#     filters = i
#     lineResult = []
#     for j in range(30, 31, 30):
#         dim = j
#         sum = 0
#         qtd = 3
#         print("Test", filters, dim)
#         for q in range(0, qtd):
#             sum += cnn_classifier(data[0], data[1], data[2], data[3], filters, 10, dim)
#         lineResult.append(sum/qtd)
#     resultMatrix.append(lineResult)

for i in resultMatrix:
    for j in i:
        print("%.4f" % round(j, 4), ";", end = '')
    print()

# Confusion matrix:
#            pred:fake  pred:real
# true:fake         28         21
# true:real          9         53

# Accuracy: 0.7297297297297297
# Precision: 0.7567567567567568
# Recall: 0.5714285714285714
# F1 score: 0.6511627906976745



# 0.6997 ;0.7027 ;0.7027 ;0.6907 ;0.6967 ;0.6847 ;0.6907 ;0.6817 ;
# 0.6757 ;0.6727 ;0.6817 ;0.6817 ;0.6697 ;0.6757 ;0.6817 ;0.6577 ;
# 0.6727 ;0.6727 ;0.6847 ;0.6787 ;0.6877 ;0.7027 ;0.6877 ;0.7027 ;
# 0.6547 ;0.6186 ;0.6396 ;0.6426 ;0.6517 ;0.6336 ;0.6426 ;0.6396 ;
# 0.6456 ;0.6486 ;0.6096 ;0.6366 ;0.6456 ;0.6426 ;0.6517 ;0.6366 ;
# 0.6276 ;0.6336 ;0.6246 ;0.5976 ;0.6156 ;0.6366 ;0.6126 ;0.6336 ;
# 0.5586 ;0.5586 ;0.5586 ;0.5646 ;0.5706 ;0.5586 ;0.5586 ;0.5586 ;


# 0.5586 ;0.5586 ;0.5706 ;0.6096 ;0.6096 ;0.6156 ;0.6036 ;
# 0.5586 ;0.5586 ;0.6006 ;0.6156 ;0.5856 ;0.5916 ;0.6156 ;
# 0.5586 ;0.5586 ;0.6006 ;0.6336 ;0.6156 ;0.6066 ;0.6216 ;
# 0.5586 ;0.5586 ;0.6186 ;0.6006 ;0.6126 ;0.5826 ;0.6276 ;
# 0.5586 ;0.5586 ;0.6006 ;0.5826 ;0.5826 ;0.6186 ;0.6306 ;
# 0.5586 ;0.5616 ;0.6096 ;0.5886 ;0.5976 ;0.6066 ;0.6607 ;
# 0.5586 ;0.5586 ;0.6276 ;0.5826 ;0.5856 ;0.5976 ;0.6306 ;
# 0.5586 ;0.5586 ;0.5766 ;0.6006 ;0.5946 ;0.6306 ;0.6697 ;


# 10 a 20 epocas
# 0.6757 ;0.6787 ;0.6817 ;0.6907 ;0.6967 ;0.7087 ;0.6937 ;0.6847 ;0.6937 ;0.6877 ;0.6787 ;
# 0.6907 ;0.6787 ;0.6757 ;0.6937 ;0.6697 ;0.6847 ;0.6967 ;0.6937 ;0.6877 ;0.6937 ;0.6937 ;
# 0.6877 ;0.6787 ;0.6937 ;0.6877 ;0.6787 ;0.6967 ;0.6727 ;0.6817 ;0.6937 ;0.6907 ;0.6847 ;
# 0.6697 ;0.6907 ;0.7027 ;0.6937 ;0.6757 ;0.6847 ;0.6757 ;0.6757 ;0.6847 ;0.6607 ;0.6727 ;

# 15 22 epocas
# 0.6937 ;0.6877 ;0.6937 ;0.6727 ;0.6997 ;0.6877 ;0.6937 ;
# 0.7057 ;0.6847 ;0.6697 ;0.6607 ;0.6847 ;0.7027 ;0.6937 ;
# 0.6787 ;0.7057 ;0.7027 ;0.6907 ;0.7177 ;0.6937 ;0.6517 ;
# 0.7027 ;0.6877 ;0.6997 ;0.6877 ;0.6937 ;0.6937 ;0.6727 ;

# dim 50 301
# 0.6937 ;0.7027 ;0.6877 ;0.7177 ;0.6637 ;0.6787 ;0.6787 ;
# 0.6817 ;0.7087 ;0.6997 ;0.6877 ;0.7087 ;0.6757 ;0.6727 ;
# 0.6667 ;0.6847 ;0.6697 ;0.6757 ;0.6997 ;0.6757 ;0.6967 ;
# 0.6967 ;0.7027 ;0.6877 ;0.6907 ;0.6967 ;0.6727 ;0.6577 ;
# 0.6937 ;0.7027 ;0.6907 ;0.6877 ;0.6727 ;0.6967 ;0.6937 ;
# 0.6817 ;0.6847 ;0.7237 ;0.6937 ;0.6937 ;0.6877 ;0.6907 ;