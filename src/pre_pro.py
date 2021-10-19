#%%
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import numpy as np 
#from aed import df

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE

#%% MODELOS
#xg boost
#tem que usar mlp né por causa da materia feedforward
  #nesse caso posso aplicar a validação durante o treino da rede
    #validação cruzada tb
#ainda preciso verificar treino igual foi feito no exercicio da outra materia

#aplicar um modelo em diferentes conjuntos de treino, igual ta no slide. 
#ver quais modelos de mlp, arquitetura da rede tb

#entre 0 e 1 usar logistico e nao hiperbolico
#mesmo com logistico, normalizar com medio 1 e 0

#ver a heuristica do numero de camadas intermediarios

#usar o treinador de hyperparemtros gridsearchCV

#começar escrever artigo também. ... ..


#%% PRE PROCESSAMENTO
#pq dropei alguns valores (duration), se basear em correlação 
    #remove highly correlated info; cause they mean the same

#transformar categorica em hot encode e normalizar númerica

#handle outliers# 
# #.90 

#explicar balanceamento, analisar pq to usando

#%%
df = pd.read_csv('../data/bank.csv', sep=';')

# %% PRÉ-PROCESSAMENTO DE DADOS ===================================================


# %% encode output pra transformar yes/no pra 1/0
df['y'] = LabelEncoder().fit_transform(df['y'])
df['y']

# %% checking the values in education field troca pra -1/1/2/3
df['education'].value_counts()

education_mapper = {"unknown":-1, "primary":1, "secondary":2, "tertiary":3}
df["education"] = df["education"].replace(education_mapper)
df


#%% drop duration
# data.drop(['duration', 'contact','month','day'], inplace=True, axis = 1)
df.drop(['duration'], inplace=True, axis = 1)


#%% one hot encoding
# listing down the features that has categorical data

categorial_features = ['job', 'marital', 'contact', 'month', 'poutcome']
# categorial_features = ['job', 'marital', 'poutcome']
for item in categorial_features:
    # assigning the encoded df into a new dfFrame object
    df1 = pd.get_dummies(df[item], prefix=item)
    df = df.drop(item, axis=1)
    for categorial_feature in df1.columns:
        #Set the new column in df to have corresponding df values
        df[categorial_feature] = df1[categorial_feature]


#binary features
binary_valued_features = ['default','housing', 'loan']
bin_dict = {'yes':1, 'no':0}

#Replace binary values in data using the provided dictionary
for item in binary_valued_features:
    df.replace({item:bin_dict},inplace=True)

df.head()
# %%rearrange the columns in the dataset to contain the y (target/label) at the end
cols = list(df.columns.values)
cols.pop(cols.index('y')) # pop y out of the list
df = df[cols+['y']] #Create new dataframe with columns in new order
df.describe()

#%%split data
y = df['y']
X = df.values[:, :-1] # get all columns except the last column

#smote
sm = SMOTE(random_state=2)
x_train_smo1, y_train_smo1 = sm.fit_resample(X, y.ravel())

#%%  checking SMOTE progress
print('no SMOTE')
print(df['y'].value_counts())

value = list(y_train_smo1)
print('SMOTE')
print('0', value.count(0))
print('1', value.count(1))


#%%

# spliting training and testing data #
X_train, X_test, y_train, y_test = train_test_split(x_train_smo1,y_train_smo1, test_size= 0.2, random_state=50)

#splitting in training, test and validation; 80 10 10 
X1_train, X1_test, y1_train, y1_test = train_test_split(x_train_smo1,y_train_smo1, test_size=0.1, random_state=1)

X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.125, random_state=1) # 0.25 x 0.8 = 0.2
#x_train_smo, X_val, y_train_smo, Y_val = train_test_split(x_train_smo1,y_train_smo1, test_size= 0.3, random_state=0)

#%%
# Feature scaling #normalazing?
scaler = StandardScaler()  
scaler.fit(X)   
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


#split data
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

print('end')

#%% pca?
#ctrl barra group comment

# from sklearn.decomposition import PCA

# # apply the PCA for feature reduction 
# #DONT THINK I NEED THIS 
# pca = PCA(n_components=0.95)
# pca.fit(X_train)
# PCA_X_train = pca.transform(X_train)
# PCA_X_test = pca.transform(X_test)

# X_train


# %%
