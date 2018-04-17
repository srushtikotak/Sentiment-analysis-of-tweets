#This code creates sentiment analysis model and trains it over training data ,tests over testing data and saves the model for prediction
import pyodbc
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import multiprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence 
from gensim.corpora.dictionary import Dictionary

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

# For Reproducibility
np.random.seed(1337)
# set parameters for model:
vocab_dim = 300
n_iterations = 10  
n_exposures = 5
window_size = 7
cpu_count = multiprocessing.cpu_count()
maxlen = 100
batch_size = 32
n_epoch = 2
input_length = 100

predict={0: 'train was late'}

'''
The database I used for training was a Indian Railways database. 
It was a collection of all the tweets posted to Indian Railway's Twitter account.
The tweets were given a sentiment : Positive/Negative and hence used for supervised training of the Sentiment model.
'''
#details to connect to IR (Micrsoft Azure) database

server = '----server address----'
database = '----database name----'
username = '----username----'
password = '----password----'
driver= '{ODBC Driver 13 for SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)

def import_db():
	data = pd.read_sql_query("Select Case when Sentiment = 'positive' then 1 else 0 END [Sentiment], Description [SentimentText] from [dbo].[TicketData] where Sentiment in ('negative','positive')", cnxn)
	pos_data = pd.read_sql_query("Select Case when Sentiment = 'positive' then 1 END [Sentiment], Description [SentimentText] from [dbo].[TicketData] where Sentiment in ('positive')", cnxn)
	neg_data = pd.read_sql_query("Select Case when Sentiment = 'negative' then 0 END [Sentiment], Description [SentimentText] from [dbo].[TicketData] where Sentiment in ('negative')", cnxn)
    	print 'dataset loaded with shape', data.shape   
    	print 'Positive dataset loaded with shape', pos_data.shape    
    	print 'Negative dataset loaded with shape', neg_data.shape     
	return data , pos_data , neg_data

data, pos_data , neg_data = import_db()
#print data 
#print pos_data
#print neg_data

'''
The 3 functions below filters the tweets,i.e, removes the tweets in hindi or any other language except english,  creates two columns "tokens" and "filtered_text" which has tweets without url,hashtag and @ 
'''
def tokenize(tweet):
    try:
        tweet = unicode(tweet.decode('utf_8').lower())
        tokens = tweet.split()
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'

def filtering(tweet):
    try:
	tweet = re.sub(r"http\S+", "", tweet)
	tweet = re.sub(r"@\S+", "", tweet)
        tweet = filter(lambda t: not t.startswith('#'), tweet)
        return tweet
    except:
        return 'NC'

def postprocess(data, n=40000):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize) 
    data['filtered_text'] = data['SentimentText'].progress_map(filtering) 
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    print data.shape
    return data

data = postprocess(data)
pos_data = postprocess(pos_data)
neg_data = postprocess(neg_data)

#data_tokens are thelist of all the tokens in the "data" dataframe

data_tokens = filter(None, [next_token for next_token in data['tokens']])

print('Training a Word2vec model...')
model = Word2Vec(size = vocab_dim, #size is the dimensionality of the feature vectors.
                 min_count = n_exposures, #min_count = ignore all words with total frequency lower than this.
                 window = window_size, #window is the maximum distance between the current and predicted word within a sentence.
                 workers = cpu_count, #workers = use this many worker threads to train the model (=faster training with multicore machines).
                 iter = n_iterations) #iter = number of iterations (epochs) over the corpus. Default is 5.
model.build_vocab(data_tokens)
model.train(data_tokens , total_examples=model.corpus_count, epochs=model.iter)
#total_examples (count of sentences)

#saving word2vec model
model.save('w2v_model')
print('w2v_model saved ')

#loading word2vec model
model = Word2Vec.load('w2v_model')  

training_set_x = [pos_data['SentimentText'].head(5000), neg_data['SentimentText'].head(5000)]
x_train = pd.concat(training_set_x, ignore_index = True)
#print x_train
print x_train.shape

training_set_y = [pos_data['Sentiment'].head(5000), neg_data['Sentiment'].head(5000)]
y_train = pd.concat(training_set_y , ignore_index = True)
#print y_train
print y_train.shape

x_train = x_train.to_dict()

testing_set_x = [pos_data['SentimentText'].tail(938), neg_data['SentimentText'].tail(938)]
x_test = pd.concat(testing_set_x, ignore_index = True)
#print x_test
print x_test.shape

testing_set_y = [pos_data['Sentiment'].tail(938), neg_data['Sentiment'].tail(938)]
y_test = pd.concat(testing_set_y , ignore_index = True)
#print y_test
print y_test.shape

x_test = x_test.to_dict()

''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
'''
def create_dictionaries(train = None,
                        test = None,
			predict = None,
                        model = None):
    if (train is not None) and (model is not None) and (test is not None) and (predict is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(data ):
            ''' Words become integers
            '''
            for key in data.keys():
                txt = data[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data[key] = new_txt
            return data

        train = parse_dataset(train )
        test = parse_dataset(test )
        predict = parse_dataset(predict )
        return w2indx, w2vec, train, test , predict
    else:
        print('No data provided...')

print('Transform the Data...')
index_dict, word_vectors, x_train, x_test ,predict = create_dictionaries(train = x_train,
                                                            test = x_test, predict = predict ,
                                                            model = model)


print('Dictionaries created')

#word vectors used as embedding layer weights
print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  
embedding_weights = np.zeros((n_symbols, vocab_dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]

print('Creating Datesets...')
X_train = x_train.values()
X_test = x_test.values()

print("Pad sequences (samples maxlen time)")
X_train = sequence.pad_sequences(X_train, maxlen = maxlen)
X_test = sequence.pad_sequences(X_test, maxlen = maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert labels to Numpy Sets...')
y_train = np.array(y_train)
y_test = np.array(y_test)

#to know more about the model and it's parameters refer the model_parameters document.

print('Defining a Simple Sequential Keras Model...')
model = Sequential()
model.add(Embedding(output_dim = vocab_dim,
                    input_dim = n_symbols,
                    mask_zero = True,
                    weights = [embedding_weights],
                    input_length = input_length))
model.add(Dropout(0.2))
#model.add(LSTM(vocab_dim,return_sequences=True))
model.add(LSTM(vocab_dim))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

print('Compiling the Model...')
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

print("Train...")
model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = n_epoch,
          validation_data = (X_test, y_test),
          shuffle = True,verbose=2)

print("Evaluate...")
score = model.evaluate(X_test, y_test,
                       batch_size = batch_size , verbose=0)

print("Accuracy: %.2f%%" % (score[1]*100))

#saving the above model
model_json = model.to_json()
with open("lstm_model.json", "w") as json_file:
	json_file.write(model_json)
# saving weights to HDF5
model.save_weights("lstm_model.h5")
print("Saved lstm_model to disk")


predict_text = predict.values()
predict_text = sequence.pad_sequences(predict_text, maxlen = maxlen)

prediction = model.predict(predict_text)
print(prediction)



