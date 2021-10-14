#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
import warnings
import random

# %% Leitura de dados ===================================================
missing_values = ["unknown"]
df = pd.read_csv('../data/bank.csv', sep=';', na_values = missing_values)
df.head()
plt.style.use('ggplot')
colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

# %% Verificando dados faltantes ===================================================
#ver o que fazer ainda

df.info()
df.describe()
print(df.isnull().sum())

# %% Analise exploratória de dados ===================================================

#ordinal and nominal
cat = df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
         'month', 'poutcome', 'y']].copy()

#cat.info()

#discrete and continuous
num = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].copy()


# %% Distribuição ===================================================

x = df['y'].value_counts().to_list()
colors = ['#D1345B','#34D1BF']
labels = ["Nao", "Sim"]
plt.title('Realizou o deposito a prazo?')
plt.pie(x, labels=labels, autopct="%1.2f%%", colors=colors[::-1], explode=[0, 0.1])

#dataset desbalanceado 
#aplicar SMOTE

# %% Outliers ===================================================
for atri in num:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=num,x=num[atri],color=random.choice(colors));

#%%
ax = num[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].plot(kind='box', title='boxplot', showmeans=True)
plt.yscale('log')
plt.show()
#bastante outlier em duration e pdays, ver oq fazer

# %% Distribuição de númericos ===================================================

for atri in num: 
    plt.rc('lines', mew=0, lw=0)
    plt.rc('axes', edgecolor='none', grid=False)
    plt.title(f"Distribuição de {atri}", fontdict={'fontsize': 14})
    plt.hist(num[atri], color=random.choice(colors), align='mid')
    plt.show()

# %% Heatmap numéricos ===================================================
correlation = pd.DataFrame(df).corr()
cm = sns.light_palette('red', as_cmap=True)
plt.pcolor(correlation, cmap=cm)
plt.yticks(np.arange(0.5, len(dataf.index), 1), dataf.index)
plt.xticks(np.arange(0.5, len(dataf.columns), 1), dataf.columns)
plt.show()
