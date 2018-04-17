import falcon
import json

import multiprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import model_from_json
from keras.preprocessing import sequence

import gensim
from gensim.models.word2vec import Word2Vec 

from gensim.corpora.dictionary import Dictionary
vocab_dim = 300
n_iterations = 10  
n_exposures = 10
window_size = 7
cpu_count = multiprocessing.cpu_count()
maxlen = 100
batch_size = 32
n_epoch = 2
input_length = 100

class Predict_model(object):
    def on_get(self, req, resp):
	#save the tweet sent as query string in description
	description = req.get_param('text')
	print description
	
	predict={0: ' ' + description}

	''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the predict Dictionary
	'''
	def create_dictionaries(predict = None,
                        model = None):
    
    		if (model is not None) and (predict is not None):
       	 		gensim_dict = Dictionary()
        		gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        		w2indx = {v: k+1 for k, v in gensim_dict.items()}
        		w2vec = {word: model[word] for word in w2indx.keys()}

        		def parse_dataset(data ):
         		 #Words become integers
           
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
        		predict = parse_dataset(predict )
       			return w2indx, w2vec, predict
    		else:
        		print('No data provided...')

	#loading word2vec model
	model = Word2Vec.load('w2v_model')  
	print('Transform the Data...')

	index_dict, word_vectors, predict= create_dictionaries(predict = predict , model = model)

	print('Dictionaries created')

	predict_text = predict.values()

	#padding the input for maxlen
	predict_text = sequence.pad_sequences(predict_text, maxlen = maxlen)

		
	#loaded_model  : loads the saved lstm_model 
	json_file = open('lstm_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("lstm_model.h5")
	print("Loaded lstm model from disk")

	prediction = loaded_model.predict(predict_text)
	print(prediction)

	prediction = int(round(prediction))
	if (prediction == 0):
		prediction = 'Negative'
	else:
		prediction = 'Positive'

        resp.body = json.dumps(prediction)


class Response_(object):
    def on_get(self, req, resp):

	tweet = req.get_param('text')
	print tweet
	#static response 
	resp.body = json.dumps("This is a reply")

