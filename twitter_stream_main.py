
from __future__ import print_function
import tweepy
import json
import re
from dateutil import parser
import urllib2

WORDS = ['#Shahrukh','Hrithik'] #Any words you wish to tags(It can be a hashtag, username or any word)
 
CONSUMER_KEY = "Your Consumer Key"
CONSUMER_SECRET = "Your Consumer Secret"
ACCESS_TOKEN = "Your Access Token"
ACCESS_TOKEN_SECRET = "Your Access Token Secret"

class StreamListener(tweepy.StreamListener):    
    #This is a class provided by tweepy to access the Twitter Streaming API. 
 
    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")
 
    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        try:
           # Decode the JSON from Twitter
            datajson = json.loads(data)
            
            #grab the wanted data from the Tweet
            text = datajson['text']
            screen_name = datajson['user']['screen_name']
            tweet_id = datajson['id']
            created_at = parser.parse(datajson['created_at']) 
            
            print("Tweet : " + text )
	    print("By : " + screen_name )
	    print("Tweet id : " + str(tweet_id))
	    tweet = urllib2.quote(text)
	    #print (" " + tweet)
#the tweet is passed as a query string to the falcon api. Prediction is the Sentiment received from the api.	    
	    prediction = json.load(urllib2.urlopen("http://localhost:8000/things?text="+tweet))
            print("Sentiment :  " + prediction)
#the tweet is passed as a query string to the falcon api. reply is the static response received from the api.	    
	    reply = json.load(urllib2.urlopen("http://localhost:8000/response?text="+tweet))
	    print(" Response : " + reply)
	    print("\n " )
#the response in "reply" parameter is sent to the tweet_id of the tweet sent
	    api.update_status("@" + screen_name + " " + reply , in_reply_to_status_id = tweet_id)
        except Exception as e:
           print(e)


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True)) 
streamer = tweepy.Stream(auth=auth, listener=listener)
streamer.filter(track=WORDS)

