from sklearn import preprocessing
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split

def Normalization(data, train_data_length):
    data_column = data.shape[1]
    min_max_scaler = preprocessing.MinMaxScaler()
    NL_data = min_max_scaler.fit_transform(data)
    NL_input_data = NL_data[:, :data_column - 1]
    NL_input_data = NL_input_data.reshape((-1, 1, data_column - 1))  # 三维
    # NL_input_data = NL_input_data.reshape((-1, data_column - 1)) # 二维
    NL_output_data = NL_data[:, -1]

    # 训练集、测试集分开
    NL_train_X_data = NL_input_data[0:train_data_length]
    print("NL_train_X_data.shape: ", NL_train_X_data.shape)
    NL_train_Y_data = NL_output_data[0:train_data_length]
    NL_test_X_data = NL_input_data[train_data_length:]
    print("NL_test_X_data.shape: ", NL_test_X_data.shape)
    NL_test_Y_data = NL_output_data[train_data_length:]
    print("NL_test_Y_data.shape: ", NL_test_Y_data.shape)
    return NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data
def Anti_Normalization(data, train_data_length, yhat):
    data_column = data.shape[1]
    min_max_scaler = preprocessing.MinMaxScaler()
    NL_data = min_max_scaler.fit_transform(data)
    NL_input_data = NL_data[:, :data_column - 1]
    NL_input_data = NL_input_data.reshape((-1, 1, data_column - 1))  # 三维
    NL_test_X_data = NL_input_data[train_data_length:]
    print("NL_test_X_data.shape: ", NL_test_X_data.shape)
    NL_test_X_data = NL_test_X_data.reshape(-1, data_column - 1)
    NL_test_yhat = np.hstack((NL_test_X_data, yhat))
    print("NL_test_yhat.shape: ", NL_test_yhat.shape)
    test_yhat = min_max_scaler.inverse_transform(NL_test_yhat)
    yhat = test_yhat[:, -1]
    yhat = yhat.reshape(yhat.shape[0], 1)
    return yhat
def Normalization_lab(data):                                      
    min_max_scaler = preprocessing.MinMaxScaler()
    NL_data = min_max_scaler.fit_transform(data)
    print(NL_data.shape)
    return NL_data
def Anti_Normalization_lab(data, NL_test_X_data, yhat, logGas):
    min_max_scaler = preprocessing.MinMaxScaler()
    NL_data = min_max_scaler.fit_transform(data)
    NL_test_yhat_logGas = np.hstack((NL_test_X_data, yhat, logGas))
    test_yhat_logGas = min_max_scaler.inverse_transform(NL_test_yhat_logGas)
    yhat = test_yhat_logGas[:, -2]
    yhat = yhat.reshape(-1, 1)
    #print("yhat:", yhat)
    return yhat

