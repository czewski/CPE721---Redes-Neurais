#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

#%%
missing_values = ["unknown"]
df = pd.read_csv('../data/bank.csv', sep=';', na_values = missing_values)
df.head()

# %% missing values #ver o que fazer ainda
df.info()
df.describe()
print(df.isnull().sum())

# %% análise dos dados

#ordinal and nominal
cat = df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
         'month', 'poutcome', 'y']].copy()

#cat.info()

#discrete and continuous
num = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].copy()

plt.style.use('ggplot')

# %% verificar distribuoção de sim

x = df['y'].value_counts().to_list()
colors = ['#D1345B','#34D1BF']
labels = ["Nao", "Sim"]
plt.title('Realizou o deposito a prazo?')
plt.pie(x, labels=labels, autopct="%1.2f%%", colors=colors[::-1], explode=[0, 0.1])

#dataset desbalanceado 
#aplicar SMOTE

# %% verificar outliers


