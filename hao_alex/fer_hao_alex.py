import numpy as np
import matplotlib.pyplot as plt
import time
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2


start_time = time.time()

K.set_image_dim_ordering('th')

# fix random seed for reproducibility (causes random function to be predictable)
seed = 1337
np.random.seed(seed)

# load the data
X_train = np.load('../data_store/fer_X_train.npy')
X_test = np.load('../data_store/fer_X_test.npy')
y_train = np.load('../data_store/fer_y_train.npy')
y_test = np.load('../data_store/fer_y_test.npy')

# reshape data to be [samples][layers][width][height]
print X_train.shape
X_train = X_train.reshape(X_train.shape[0], 1, 48, 48).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 48, 48).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train.astype(int), 7)
y_test = np_utils.to_categorical(y_test.astype(int), 7)
num_classes = y_test.shape[1]

# define cnn model
def cnn_model():
	model = Sequential()
	
	model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(1, 48, 48)))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(64, 48, 48)))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Flatten())
	# Note: Keras does automatic shape inference.
	model.add(Dense(3072))
	model.add(Activation('relu'))

	model.add(Dense(num_classes, activity_regularizer=l2(0.01)))
	model.add(Activation('linear'))

	model.compile(loss='', optimizer='sgd', metrics=['accuracy'])
	return model

# build the model
model = cnn_model()
# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=100, verbose=2)
# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

# print results
#print model.summary()
#print model.get_weights()
print model.count_params()
print model.metrics_names
print "CNN Error: %.2f%%" % (100 - scores[1] * 100)
print "--- %s seconds ---" % (time.time() - start_time)

model.save("./fer_hao_alex")
