#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from xgboost import XGBClassifier

from pre_pro import PCA_X_test, PCA_X_train, X_train, X_test, y_train, y_test  
from pre_pro import X1_train,y1_train, X1_test,y1_test, X1_val,y1_val

import tensorflow as tf
import os #disable logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from ann_visualizer.visualize import ann_viz;

#%% KERAS ANN MLP
#smote values arent checked with correlation

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units= 44, activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units= 10, activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
ann.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics= ['accuracy'])  #tentar softmax

#earlystop
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')  #testar patiance

ann_history = ann.fit(PCA_X_train,y_train, 
                      batch_size= 32, epochs= 50, validation_split=0.2, callbacks=[early_stop])
#validation_data=(X1_val, y1_val)

#y_pred = ann.predict(X_test) #wtf is supposed to be here
#y_pred = [round(x[0]) for x in y_pred]
ann.summary()

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


#%%
mlp=MLPClassifier(
   hidden_layer_sizes=(500,2000,500),
   activation="logistic",
   solver='adam', #sgd
   max_iter=300, 
   validation_fraction=0.1, 
   
   learning_rate='constant', 
   epsilon=1e-8,  #1e-8
   random_state=5,      #reprodução
   early_stopping=True, #parada antes
   tol=0.000010,        #default tol=0.000100
   n_iter_no_change=50, #10
   verbose=True         #aviso atenção
   )
#mlp.fit(PCA_X_train,y_train)
mlp.fit(X_train,y_train)
#print (mlp.score(PCA_X_test,y_test))
print (mlp.score(X_test, y_test))
plt.plot(mlp.loss_curve_)
plt.plot(mlp.validation_scores_)

#%% mlp com criterio de parada #antes separar em treino valid teste
max_iter = 500
mlp = MLPClassifier(hidden_layer_sizes=(1,16,1), activation='logistic', solver='adam', max_iter=max_iter)
train_errors = []
validation_errors = []
for i in range(max_iter):
     mlp.partial_fit(X1_train, y1_train, classes=np.unique(y1_train))
     y_predict = mlp.predict(X1_train) #test maybe train here
     y_predicted = mlp.predict(X1_val) #validação
     
     train_error = mean_absolute_error(y1_train, y_predict)
     train_errors.append(train_error) #test

     validation_error = mean_absolute_error(y1_val, y_predicted)
     validation_errors.append(validation_error) #validação
     
     if np.remainder(i+1,100)==0:
        print(((i+1)*100)/max_iter)
        
     if abs(train_error-validation_error)>=.1:
         break
     
#%%     
#print(mlp.score(X1_train,y1_train))
print(mlp.score(X1_train,y1_train)) #score do treino
print(mlp.score(X1_test, y1_test)) #score da validação

#figures
fig, ax = plt.subplots()
ax.plot(train_errors, lw=1)
ax.plot(validation_errors, lw=1)
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
plt.title('Treino x Validação')
plt.show()

# fig, ax = plt.subplots()
# ax.plot(X1_test,y_predict,'o-', lw=1)
# #ax.plot(M[:,0], M[:,2], 'o--', lw=1)
# ax.set_xlabel('Entrada')
# ax.set_ylabel('Saída')
# plt.title('Treino')
# plt.show()


# X_test=X_test.reshape(-1)
# y_test=y_test.reshape(-1)
# V = np.matrix([X_test,y_test,y_predicted]).transpose()
# V.sort(axis=0, kind=None, order=None)
# fig, ax = plt.subplots()
# ax.plot(V[:,0],V[:,1],'o-', lw=1)
# ax.plot(V[:,0], V[:,2], 'o--', lw=1) #predicted
# ax.set_xlabel('Entrada')
# ax.set_ylabel('Saída')
# plt.title('Validação')
# plt.show()

# X_test=X_test.reshape(-1,1)
# y_test=y_test.reshape(-1,1)

#%% BASIC MLP CLASSIFIER
# define and train an MLPClassifier named mlp on the given data
mlp = MLPClassifier(hidden_layer_sizes=(50,200,50), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp.fit(PCA_X_train, y_train)
# %%
print('Accuracy')
print(mlp.score(PCA_X_test, y_test))

# draw the confusion matrix
predict = mlp.predict(PCA_X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, predict)
fig, ax = plt.subplots(1)
ax = sns.heatmap(confusion_matrix, ax=ax, cmap=plt.cm.Blues, annot=True)
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()

# %% print the training error and MSE
print("Training error: %f" % mlp.loss_curve_[-1])
print("Training set score: %f" % mlp.score(PCA_X_train, y_train))
print("Test set score: %f" % mlp.score(PCA_X_test, y_test))
print(accuracy_score(y_test, predict))

print("MSE: %f" % mean_squared_error(y_test, predict))

# %% xgboost
model = XGBClassifier()
model.fit(PCA_X_train, y_train, verbose=False)
predictions = model.predict(PCA_X_test)
print("Erro Médio Absoluto: {:.2f}".format(mean_squared_error(predictions, y_test)))

confusion_matrix = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(1)
ax = sns.heatmap(confusion_matrix, ax=ax, cmap=plt.cm.Blues, annot=True)
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()
# %%
