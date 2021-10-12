#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

#%%
df = pd.read_csv('../data/bank.csv', sep=';')
df.head()

# %% no missing values;
df.info()
df.describe()
# %% transformar categorica em hot encode e normalizar númerica
#transformar month pra 1-12
