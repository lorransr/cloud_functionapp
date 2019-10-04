#importando bibliotecas
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
import os
#importando dados
boston_dataset = datasets.load_boston()
#definindo variaveis
bos = pd.DataFrame(boston_dataset.data)
bos.columns = boston_dataset.feature_names
X = bos[['LSTAT', 'RM']].values
y = boston_dataset.target
#fitando o modelo
regressor = LinearRegression(normalize=True)
regressor.fit(X,y)
#criando a pasta model
os.mkdir("model")
#salvando o modelo
pickle.dump(regressor,open('model/regressor.pickle','wb'))
