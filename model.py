from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping
from keras import regularizers
def LinearRegression_model1(NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data):
    model = LinearRegression()
    model.fit(NL_train_X_data, NL_train_Y_data)
    score = model.score(NL_train_X_data, NL_train_Y_data)
    ev = model.coef_  
    inter = model.intercept_
    yhat = model.predict(np.array(NL_test_X_data))
    yhat = yhat.reshape(yhat.shape[0], 1)
    return yhat, model, ev, inter
def LinearRegression_model2(Arg, DepVar):
    model = LinearRegression()
    Arg = Arg.reshape(-1, 1)
    DepVar = DepVar.reshape(-1, 1)
    model.fit(Arg, DepVar)
    score = model.score(Arg,DepVar)
    ev = model.coef_  
    inter = model.intercept_
    LR_DepVar = model.predict(Arg)
    LR_DepVar = LR_DepVar.reshape(-1, 1)
    #     print("yhat:", yhat)
    #     print("yhat_array[test_index][0]:", yhat_array[test_index][0])
    return LR_DepVar, ev, inter, score

def bp_model(NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data):
    model = Sequential()
    model.add(Dense(10,activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')  
    history = model.fit(NL_train_X_data, NL_train_Y_data, batch_size=20, epochs=200, validation_data=(NL_test_X_data, NL_test_Y_data), verbose=0)
    yhat = model.predict(np.array(NL_test_X_data), verbose=0)
    yhat = yhat.reshape(yhat.shape[0], 1)
    return yhat, model, history


def three_cnn_model(NL_train_X_data, NL_train_Y_data, NL_test_X_data,NL_test_Y_data, filter_Num):
    model = Sequential()
    model.add(Conv1D(filters=filter_Num, kernel_size=2, activation='relu',
                     strides=1, padding='same', data_format='channels_first'))
    model.add(Conv1D(filters=filter_Num, kernel_size=2, activation='relu',
                     strides=1, padding='same', data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='same',
                           data_format='channels_first'))
    model.add(Conv1D(filters=filter_Num + 32, kernel_size=2, activation='relu',
                     strides=1, padding='same', data_format='channels_first'))
    model.add(Conv1D(filters=filter_Num + 32, kernel_size=2, activation='relu',
                     strides=1, padding='same', data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='same',
                           data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(units=30, activation='sigmoid',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse',
                  metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
                  target_tensors=None)
    history = model.fit(NL_train_X_data, NL_train_Y_data, batch_size=5, epochs=20, validation_data=(NL_test_X_data, NL_test_Y_data), verbose=0)
    print('\n', model.summary())
    yhat = model.predict(np.array(NL_test_X_data), verbose=0)
    yhat = yhat.reshape(yhat.shape[0], 1)
    # print("yhat.shape:", yhat.shape)
    return yhat, model, history



def one_cnn_model(NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data, filter_Num):
    model = Sequential()
    model.add(Conv1D(filters=filter_Num, kernel_size=2, activation='relu',
                     strides=1, padding='same', data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='same',
                           data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(NL_train_X_data, NL_train_Y_data, batch_size=5, epochs=30, validation_data=(NL_test_X_data, NL_test_Y_data), verbose=0)
    yhat = model.predict(np.array(NL_test_X_data), verbose=0)
    yhat = yhat.reshape(yhat.shape[0], 1)
    return yhat, model, history


