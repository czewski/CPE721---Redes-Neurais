#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix  
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from pre_pro import  X_train, X_test, y_train, y_test, X_val, y_val
#from pre_pro import PCA_X_test, PCA_X_train,

import tensorflow as tf
import os #disable logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.backend import clear_session
from keras.metrics import confus
from ann_visualizer.visualize import ann_viz;



#%% KERAS ANN MLP
#maybe add dropout?
clear_session()

ann = tf.keras.models.Sequential(name='MLPFullyConnected')
ann.add(tf.keras.layers.InputLayer(input_shape=(44,)))
ann.add(tf.keras.layers.Dense(units= 10, activation='relu')) #testar tgh
#ann.add(tf.keras.layers.Dense(units= 10, activation='relu')) #testar tgh
ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid')) #sigmoid

optimizer = Adam()

optimizer = SGD(learning_rate=0.1,
                  
                 )# momentum=.9 decay=1e-5,

ann.compile(optimizer=optimizer,
            loss= 'mean_squared_error', 
            metrics= ['accuracy']) #adam

early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0, 
                           patience=5, 
                           verbose=0, 
                           mode='auto')  #testar patiance

ann_history = ann.fit(X_train,y_train, 
                      shuffle=True, 
                      use_multiprocessing=True, #validation_split=0.1,
                      batch_size= 32, 
                      epochs= 200, 
                      validation_data = (X_val, y_val),  
                      callbacks=[early_stop])

scores = ann.evaluate(X_test, y_test, batch_size=32)

#Loss
loss_train = ann_history.history['loss']
loss_val = ann_history.history['val_loss']
epochs = range(1,len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
#Acuracia
print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))

ann.summary()
ann.get_config()
loss_train = ann_history.history['accuracy']
loss_val = ann_history.history['val_accuracy']
epochs = range(1,len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% model visualization
from keras.utils.vis_utils import plot_model
plot_model(ann, to_file='model.png')
#%%confusion matrix
y_pred = ann.predict_classes(X_test)
#print(y_pred)
cma = confusion_matrix(y_test, y_pred) #labels='1,2'
print(cma)

#%%error histogram for train, test, validation

sns.histplot(data=ann_history.history['loss'], kde=True, log_scale=True, element='step', fill=False)

# %%
