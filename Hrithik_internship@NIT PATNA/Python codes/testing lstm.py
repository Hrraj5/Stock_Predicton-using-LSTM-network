
# coding: utf-8

# In[4]:


from sklearn.model_selection import StratifiedKFold,KFold
from keras.preprocessing.text import one_hot
#from keras.preprocessing.text import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import History
from numpy import zeros
from keras.layers.embeddings import Embedding
import pandas as pd
import numpy as np
from prf1 import precision, recall, f1_score
from keras.utils.vis_utils import plot_model
#import pydot
#import graphviz
from keras import backend as K
from keras.preprocessing import sequence
from keras.layers import LSTM
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
#from confusionMetrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from prf1 import precision, recall, f1_score
from keras.utils import np_utils
from imblearn.over_sampling import SMOTE
#######Loading CSV file using Pandas
 
df=pd.read_excel('airtelpcj.xlsx')


#print((df.info()))

df1=df['message']
label =df['Label']

#x_val, y_val = sm.fit_sample(x_val1, y_val1)
#print(label)
#Y=np_utils.to_categorical(Y_old)
#print(df1.info())
tk=Tokenizer()
tk.fit_on_texts(df1)
index=tk.word_index
#print(index)
x = tk.texts_to_sequences(df1)
#print (x)
seed = 7
numpy.random.seed(seed)
##encoded_doc = tk.texts_to_matrix(df, mode='count')
##print (encoded_doc)
##max_length=15
##padded_docs = sequence.pad_sequences(encoded_doc, maxlen=max_length, padding='pre')
##print (padded_docs)
vocab_size = len(index)


#print(vocab_size)
encoded_docs=[one_hot(d,vocab_size) for d in df1] 
#print (hello)
##############################Padding###############
max_length=50
padded_docs = sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print padded_docs
#####################Encoded label#######
encoder = LabelEncoder()
encoder.fit(label)
encoded_label = encoder.transform(label)
#print(encoded_label)
labels = np_utils.to_categorical(encoded_label)
#print(labels)
#print('Shape of data tensor:', x_train.shape)
#print('Shape of label tensor:', labels.shape)



#x_train1, x_val1, y_train1, y_val1 = check_X_y(padded_docs, labels, accept_sparse="csc", dtype=np.float32, multi_output=True)
#x_train1, x_val1, y_train1, y_val1=train_test_split(padded_docs,labels, test_size=0.25,random_state=42)
#print(x_train1.shape)
#print(x_val1.shape)
#print(y_train1.shape)
#print(y_val1.shape)
#sm = SMOTE(kind='regular')
#x_train, y_train = sm.fit_sample(x_train1,y_train1)
#x_val,y_val=sm.fit_sample(x_val1,y_val1)
 
embeddings_index = {}
f = open('glove.6B.100d.txt',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
 
print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))
 
# In[10]:
embedding_matrix =zeros((vocab_size+ 1, 100))
for word, i in tk.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector   
print ("Embedding Matrix")
 
 
 
#############embedding layers#########################
 
history=History()
predictions_train=[]
predictions_test=[]
filter_sizes = [3,4,5]
fold_training=numpy.zeros(shape=75)
fold_test=numpy.zeros(shape=75)
##embedding_vecor_length =32
model = Sequential()
model.add(Embedding(vocab_size+1, 100,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False))
##model.add(Embedding(vocab_size,8, input_length=embedding_vecor_length))
kfold = KFold(n_splits=20, shuffle=True, random_state=7)
cvscores = []
ith=1
for train,test in kfold.split(padded_docs,labels):
    #print('Fold=',ith)
    #ith=ith+1
    #print(train,test)
   # X_train, X_test = padded_docs[train], padded_docs[test]
    #Y_train, Y_test = labels[train], labels[test]
    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(200,return_sequences=True))
    model.add(LSTM(200,return_sequences=True))
    model.add(LSTM(200,return_sequences=True,recurrent_dropout=0.2))
    model.add(LSTM(100))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['acc', precision, recall, f1_score])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision,recall,f1_score])
    print(model.summary())
    history=model.fit(padded_docs[train], labels[train],validation_data=(padded_docs[test],labels[test]),epochs=75, batch_size=20,verbose=2, callbacks=[history])
    predictions = model.predict(padded_docs[test])
    b=np.zeros_like(predictions)
    b[np.arange(len(predictions)), predictions.argmax(1)]=1
    print( metrics.classification_report(labels[test],b))

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
fold_training=numpy.divide(fold_training,20)
print(fold_training)
fold_test=numpy.divide(fold_test,20)
print(fold_test)
accuracy = model.evaluate(padded_docs[test],labels[test], verbose=0)
print('Accuracy: %f' % (accuracy*100))
    
    
                               
                               
    
'''history=History()
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.ylabel('precision')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()'''

