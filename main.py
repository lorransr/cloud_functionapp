from flask import Flask, request, json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

model_file = 'model/regressor.pickle'

def make_prediction(request):
    # carregando modelo
    model = pickle.load(open(model_file,"rb"))
    #recebendo o Post
    data =  request.get_json(force=True)
    x = np.array(data["data"]).reshape(1,2)
    #fazendo a predicao
    prediction = model.predict(x)
    result = {"result":float(prediction)}

    return json.dumps(result)

if __name__ == '__main__':
    from flask import Flask, request
    app = Flask(__name__)
    app.route('/',methods=["POST"])(lambda: make_prediction(request))
    app.run(debug=True)
