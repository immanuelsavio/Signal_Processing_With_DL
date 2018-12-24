from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import numpy as np

seed = 9
np.random.seed(seed)

dataset = np.loadtxt('SC4001E0_features_(30s_66f)_58_A_D_F_66_Sen1-Replaced.csv', delimiter=',', skiprows=1)

X = dataset[:,0:66]
Y = dataset[:,66]


(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.20, random_state=seed)
# create the model
model = Sequential()
model.add(Dense(66, input_dim=66, init='uniform', activation='relu'))
model.add(Dense(500, init='uniform', activation='softmax'))
model.add(Dense(250, init='uniform', activation='relu'))
model.add(Dense(50, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=100, batch_size=5)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print ("Accuracy: %.2f%%" %(scores[1]*100)) 

