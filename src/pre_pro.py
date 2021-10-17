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

#xg boost
#tem que usar mlp né por causa da materia feedforward
#ainda preciso verificar treino igual foi feito no exercicio da outra materia
#começar escrever artigo também. ... ..

#%%
df = pd.read_csv('../data/bank.csv', sep=';')

# %% PRÉ-PROCESSAMENTO DE DADOS ===================================================

# %% transformar categorica em hot encode e normalizar númerica

#handle outliers# 
# #.90 

#remove highly correlated info; cause they mean the same

#normalize data

#balancear ainda, usar smote

# %% encode output
df['y'] = LabelEncoder().fit_transform(df['y'])
df['y']

# %% checking the values in education field
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


# %%rearrange the columns in the dataset to contain the y (target/label) at the end
cols = list(df.columns.values)
cols.pop(cols.index('y')) # pop y out of the list
df = df[cols+['y']] #Create new dataframe with columns in new order
df.describe()


#%%split data
y = df['y']
X = df.values[:, :-1] # get all columns except the last column

# spliting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)


#%%
# Feature scaling #normalazing?
scaler = StandardScaler()  
scaler.fit(X)   
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


#split data
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#%% pca?

from sklearn.decomposition import PCA

# apply the PCA for feature for feature reduction
pca = PCA(n_components=0.95)
pca.fit(X_train)
PCA_X_train = pca.transform(X_train)
PCA_X_test = pca.transform(X_test)

X_train


# %%
