from falcon_api_predict_response import *
	
app = falcon.API()

#things calls the Predict_model class in falcon_api_predict_response.py
things = Predict_model()
#response calls the Response_ class in falcon_api_predict_response.py
response = Response_()

# things will handle all requests to the '/things' URL path
app.add_route('/things', things)
# response will handle all requests to the '/response' URL path
app.add_route('/response', response)

