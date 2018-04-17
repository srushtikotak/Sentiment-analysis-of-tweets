## Sentiment-analysis-of-tweets

#### Problem Statement : Build a system which collects the live tweets posted to ABC's twitter account then :
1. Find the Sentiment of that tweet.
2. Based on the Sentiment generate an appropriate reply.
3. Reply back to that tweet.

#### Work Flow:

1. Collect the live streaming tweets using Python’s library : Tweepy.
2. From each tweet, extract the necessary parameters, i.e,  Text(actual tweet), user_id, screen_name, time , tweet_id.
3. Pass the text to the Falcon web API which we created  as a Query string.
4. The Falcon API has a class which loads our trained Sentiment analysis model and gives the sentiment of that tweet.
5. Based on the Sentiment , the tweet can be sent to the ChatBot, again by calling a falcon web API , which will return the response to that tweet.
(For now , our API returns a static response)
6. Store the response and reply back to the tweet with that response.

#### Following is the list of codes needed:
1. **twitter_stream_main.py** : 
This code collects the Streaming data, extracts the necessary parameter , sends the tweet to falcon api , prints the sentiment , sends the tweet to the api which returns the response and then replies to that tweet.
PS : For more details, refer : [Retrieving and storing tweets using Python and MySQL](https://github.com/srushtikotak/Retrieving-and-storing-tweets-using-Python-and-MySQL.git)

2. **falcon_api_predict_response.py** : 
This code contains two classes : Predict_model and Response_ .  Predict_model : (Step 4 from workflow) and  Response_ : (Step 5 from workflow).

3. **falcon_api.py** :
Imports the classes from falcon_api_predict_response.py and creates a route for each.

4. **Sentiment_train.py** :  (This code is only used for training the sentiment analysis model) 
This is the main code which trains the Sentiment analysis model and saves it for prediction.

#### Prerequisites:(List of libraries needed)
1. Python 2.7/3.6
2. pip/pip3 (used to install following libraries by using : sudo pip install lib_name )
3. tweepy (Tweepy is open-sourced, hosted on GitHub and enables Python to communicate with Twitter platform and use its API)
4. json (JSON (JavaScript Object Notation) is a lightweight data-interchange format.)
5. tensorflow (An open-source software library for Machine Intelligence.)
6. keras==2.0.4(Keras is a high-level neural networks API, written in Python and capable of running on top of either TensorFlow, CNTK or Theano)
7. pandas (pandas is a software library written for the Python programming language for data manipulation and analysis)
8. numpy (NumPy is the fundamental package for scientific computing with Python)
9. pyodbc  (pyodbc is an open source Python module that makes accessing ODBC databases simple)
10. gensim (Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora)
11. tqdm
12. falcon (Falcon is a very fast, very minimal Python web framework for building microservices, app backends, and higher-level frameworks.)
13. gunicorn (Gunicorn 'Green Unicorn' is a Python WSGI HTTP Server for UNIX. It's a pre-fork worker model)
(PS : I am using Ubuntu 16.04)
