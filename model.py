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
    ev = model.coef_  # 回归系数
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
    ev = model.coef_  # 回归系数
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
    model.compile(optimizer='adam', loss='mse')  # 定义优化方法为随机梯度下降，损失函数为mse
    # x->训练集,y——>bia标签,epochs=10000训练的次数,validation_data=(test_x,test_y)——>验证集
    history = model.fit(NL_train_X_data, NL_train_Y_data, batch_size=20, epochs=200, validation_data=(NL_test_X_data, NL_test_Y_data), verbose=0)
    # print('\n', model.summary())
    yhat = model.predict(np.array(NL_test_X_data), verbose=0)
    yhat = yhat.reshape(yhat.shape[0], 1)
    # print("yhat.shape:", yhat.shape)
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
    # Dense执行以下操作：output=activation（dot（input，kernel）+bias），
    # 其中,activation是激活函数，kernel是由层创建的权重矩阵，bias是由层创建的偏移向量（仅当use_bias为True时适用）。
    # 2D 输入：(batch_size, input_dim)；对应 2D 输出：(batch_size, units)
    model.add(Dense(units=30, activation='sigmoid',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    # 因为要预测下一个时间步的值，因此units设置为1
    model.add(Dense(units=1))
    # 配置模型
    model.compile(optimizer='adam', loss='mse',
                  metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
                  target_tensors=None)
    # X为输入数据，y为数据标签；batch_size：每次梯度更新的样本数，默认为32。
    # verbose: 0,1,2. 0=训练过程无输出，1=显示训练过程进度条，2=每训练一个epoch打印一次信息
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
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # Dense执行以下操作：output=activation（dot（input，kernel）+bias），
    # 其中,activation是激活函数，kernel是由层创建的权重矩阵，bias是由层创建的偏移向量（仅当use_bias为True时适用）。
    # 2D 输入：(batch_size, input_dim)；对应 2D 输出：(batch_size, units)
    # model.add(Dense(units=15, activation='relu'))
    # 因为要预测下一个时间步的值，因此units设置为1
    model.add(Dense(units=1, activation='relu'))
    # 配置模型
    model.compile(optimizer='adam', loss='mse')
    # X为输入数据，y为数据标签；batch_size：每次梯度更新的样本数，默认为32。
    # verbose: 0,1,2. 0=训练过程无输出，1=显示训练过程进度条，2=每训练一个epoch打印一次信息
    history = model.fit(NL_train_X_data, NL_train_Y_data, batch_size=5, epochs=30, validation_data=(NL_test_X_data, NL_test_Y_data), verbose=0)
    yhat = model.predict(np.array(NL_test_X_data), verbose=0)
    yhat = yhat.reshape(yhat.shape[0], 1)
    #将test和yhat放在一起进行还原 `
    return yhat, model, history


