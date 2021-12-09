import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import openpyxl
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import tree
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import random
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
from model import one_cnn_model, three_cnn_model, bp_model, LinearRegression_model1, LinearRegression_model2
from DataSet import dataset_lab
from Normalization import Normalization, Anti_Normalization, Normalization_lab, Anti_Normalization_lab

b = 6
a=pd.DataFrame(data[:,0:-1]).corr()
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(9, 9))
sns.heatmap(a, annot=True, vmax=1, square=True, cmap="Blues", xticklabels=False, yticklabels=False, annot_kws={'size':12})
plt.show()
data = np.loadtxt(r"C:\Users\Administrator\Projects\LongMaXi\unShuffle_data")
shuffle_data = data[:,[0,1,2, 4, 5]]
print(shuffle_data.shape)
np.random.shuffle(shuffle_data)
print(shuffle_data.shape)
model = LinearRegression()
shuffle_RHOB = shuffle_data[:, 2].reshape(-1, 1)
shuffle_GAS = shuffle_data[:, 3].reshape(-1, 1)
model.fit(shuffle_RHOB, shuffle_GAS)
score = model.score(shuffle_RHOB, shuffle_GAS)
ev = model.coef_  # 回归系数
inter = model.intercept_
shuffle_LR_GAS = model.predict(shuffle_RHOB).reshape(-1, 1)
print("ev:", ev)
print("inter:", inter)
Onefil_data = []
b = 1.6
for i in range(shuffle_data.shape[0]):
    if(-b< (shuffle_GAS[i] - shuffle_LR_GAS[i]) < b):
        Onefil_data.append(shuffle_data[i])
Onefil_data = np.array(Onefil_data)
print(Onefil_data.shape)
Onefil_NL_data = Normalization_lab(Onefil_data)
a=pd.DataFrame(Onefil_NL_data[:,0:-1]).corr()
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(9, 9))
sns.heatmap(a, annot=True, vmax=1, square=True, cmap="Blues", xticklabels=False, yticklabels=False, annot_kws={'size':12})
plt.show()
loo = LeaveOneOut()
Onefil_cnn =[]
Onefil_NL_input_data = Onefil_NL_data[:, 0: 3].reshape(-1, 1, 3)
Onefil_NL_output_data = Onefil_NL_data[:, 3].reshape(-1, 1, 1)
loo.get_n_splits(Onefil_NL_input_data)
for train_index, test_index in loo.split(Onefil_NL_input_data):
    NL_train_X_data, NL_test_X_data = Onefil_NL_input_data[train_index], Onefil_NL_input_data[test_index]
    NL_train_Y_data, NL_test_Y_data = Onefil_NL_output_data[train_index], Onefil_NL_output_data[test_index]
    Onefil_yhat, model, history = three_cnn_model(NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data, 32)
    NL_test_X_data = NL_test_X_data.reshape(1,-1)
    Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat, Onefil_NL_data[2,-1].reshape(-1, 1))
    Onefil_cnn.append(Onefil_yhat)
Onefil_cnn = np.array(Onefil_cnn)
Onefil_NL_input_data = Onefil_NL_data[:, 0: 3].reshape(-1, 1, 3)
Onefil_NL_output_data = Onefil_NL_data[:, 3].reshape(-1, 1, 1)
loo.get_n_splits(Onefil_NL_input_data)
Onefil_bp =[]
for train_index, test_index in loo.split(Onefil_NL_input_data):
    NL_train_X_data, NL_test_X_data = Onefil_NL_input_data[train_index], Onefil_NL_input_data[test_index]
    NL_train_Y_data, NL_test_Y_data = Onefil_NL_output_data[train_index], Onefil_NL_output_data[test_index]
    Onefil_yhat, model, history = bp_model(NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data)
    NL_test_X_data = NL_test_X_data.reshape(1,-1)
    Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat,Onefil_NL_data[2,-1].reshape(-1, 1))
    Onefil_bp.append(Onefil_yhat)
Onefil_bp= np.array(Onefil_bp).reshape(-1, 1)
Onefil_NL_input_data = Onefil_NL_data[:, 0:3]
print(Onefil_NL_input_data.shape)
Onefil_NL_output_data = Onefil_NL_data[:,3]
x_len = Onefil_NL_input_data.shape[1]
loo = LeaveOneOut()
loo.get_n_splits(Onefil_NL_input_data)
Onefil_rf =[]
for train_index, test_index in loo.split(Onefil_NL_input_data):
    NL_train_X_data, NL_test_X_data = Onefil_NL_input_data[train_index], Onefil_NL_input_data[test_index]
    NL_train_Y_data, NL_test_Y_data = Onefil_NL_output_data[train_index], Onefil_NL_output_data[test_index]
    NL_train_X_data, NL_test_X_data = NL_train_X_data.reshape(-1, x_len), NL_test_X_data.reshape(-1, x_len)
    NL_test_X_data = NL_test_X_data.reshape(-1, x_len)
    rfr = RandomForestRegressor(n_estimators = 128, min_samples_leaf = 14, random_state=13)
    Onefil_yhat = rfr.fit(NL_train_X_data,NL_train_Y_data).predict(NL_test_X_data).reshape(-1, 1)
    Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat, Onefil_NL_data[2,-1].reshape(-1, 1))
    Onefil_rf.append(Onefil_yhat)
Onefil_rf= np.array(Onefil_rf).reshape(-1, 1)
print(Onefil_rf.shape)
loo = LeaveOneOut()
loo.get_n_splits(Onefil_NL_input_data)
Onefil_dt =[]
for train_index, test_index in loo.split(Onefil_NL_input_data):
    NL_train_X_data, NL_test_X_data = Onefil_NL_input_data[train_index], Onefil_NL_input_data[test_index]
    NL_train_Y_data, NL_test_Y_data = Onefil_NL_output_data[train_index], Onefil_NL_output_data[test_index]
    NL_train_X_data, NL_test_X_data = NL_train_X_data.reshape(-1, x_len), NL_test_X_data.reshape(-1, x_len)
    NL_test_X_data = NL_test_X_data.reshape(-1, x_len)
    dtr = tree.DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 20, random_state=13)
    Onefil_yhat = dtr.fit(NL_train_X_data, NL_train_Y_data).predict(NL_test_X_data).reshape(-1, 1)
    Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat,Onefil_NL_data[2,-1].reshape(-1, 1))
    Onefil_dt.append(Onefil_yhat)
Onefil_dt= np.array(Onefil_dt).reshape(-1, 1)
loo = LeaveOneOut()
loo.get_n_splits(Onefil_NL_input_data)
Onefil_svr =[]
for train_index, test_index in loo.split(Onefil_NL_input_data):
    NL_train_X_data, NL_test_X_data = Onefil_NL_input_data[train_index], Onefil_NL_input_data[test_index]
    NL_train_Y_data, NL_test_Y_data = Onefil_NL_output_data[train_index], Onefil_NL_output_data[test_index]
    NL_train_X_data, NL_test_X_data = NL_train_X_data.reshape(-1, x_len), NL_test_X_data.reshape(-1,x_len)
    NL_test_X_data = NL_test_X_data.reshape(-1, x_len)
    svr_rbf = SVR(kernel='rbf', C=1e2, gamma = 0.4)
    Onefil_yhat = svr_rbf.fit(NL_train_X_data, NL_train_Y_data).predict(NL_test_X_data).reshape(-1, 1)
    Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat, Onefil_NL_data[2,-1].reshape(-1, 1))
    # Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat)
    Onefil_svr.append(Onefil_yhat)
#     print("yhat:", yhat)
#     print("yhat_array[test_index][0]:", yhat_array[test_index][0])
Onefil_svr= np.array(Onefil_svr).reshape(-1, 1)
print(Onefil_svr.shape)
print(Onefil_svr)
excel = np.hstack(((Onefil_data[:,3]).reshape(-1,1),Onefil_svr, Onefil_dt, Onefil_rf, 
                  Onefil_bp.reshape(-1,1), Onefil_cnn.reshape(-1, 1)))
print(excel)
loo = LeaveOneOut()
loo.get_n_splits(Onefil_NL_input_data)
Onefil_lr =[]
for train_index, test_index in loo.split(Onefil_NL_input_data):
    NL_train_X_data, NL_test_X_data = Onefil_NL_input_data[train_index], Onefil_NL_input_data[test_index]
    NL_train_Y_data, NL_test_Y_data = Onefil_NL_output_data[train_index], Onefil_NL_output_data[test_index]

    NL_train_X_data, NL_test_X_data = NL_train_X_data.reshape(-1, x_len), NL_test_X_data.reshape(-1, x_len)
    NL_test_X_data = NL_test_X_data.reshape(-1, x_len)
    Onefil_yhat, model, score, ev  = LinearRegression_model1(NL_train_X_data, NL_train_Y_data, NL_test_X_data, NL_test_Y_data)
    Onefil_yhat = Anti_Normalization_lab(Onefil_data, NL_test_X_data, Onefil_yhat, Onefil_NL_data[2,-1].reshape(-1, 1))
    Onefil_lr.append(Onefil_yhat)
Onefil_lr= np.array(Onefil_lr).reshape(-1, 1)
print('相关系数: ', np.corrcoef(data[:,0], data[:, 4]))
print('R2: ', r2_score(Onefil_data[:,3], Onefil_svr.reshape(-1, 1)))
print('mse: ', mean_squared_error(Onefil_data[:,3], Onefil_cnn.reshape(-1, 1)))
plt.figure(figsize=(20,10))
plt.plot(Onefil_data[:,4], '-', linewidth=2)
plt.plot(Onefil_lr, '-', linewidth=2)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);
plt.show
plt.figure(figsize=(20, 10))
plt.plot(Onefil_data[:,4], '-', linewidth=2)
plt.plot(Onefil_data[:,5], '-', linewidth=2)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);
plt.show
fig = plt.figure(figsize=(20,5))
plt.plot(Onefil_data[:,3], '-', linewidth=2, label='Core_value')
plt.plot(Onefil_svr, '-', linewidth=2, label='DT_predict_value')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
fig_1 = plt.figure(figsize=(20, 10))  # 作图画布大小(每循环一次，就会建立新的画布）
ax = fig_1.add_subplot( 5, 1,1)
ax.plot(Onefil_data[:,3], '-', linewidth=2)
ax.plot(Onefil_svr, '-', linewidth=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks(range(0,6,2))
ax = fig_1.add_subplot(5, 1, 2)
ax.plot(Onefil_data[:,3], '-', linewidth=2)
ax.plot(Onefil_dt, '-', linewidth=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks(range(0,6,2))
ax = fig_1.add_subplot(5, 1, 3)
ax.plot(Onefil_data[:,3], '-', linewidth=2)
ax.plot(Onefil_rf, '-', linewidth=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks(range(0,6,2))
ax = fig_1.add_subplot(5, 1, 4)
ax.plot(Onefil_data[:,3], '-', linewidth=2)
ax.plot(Onefil_bp, '-', linewidth=2)
plt.yticks(range(0,6,2))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax = fig_1.add_subplot(5, 1, 5)
ax.plot(Onefil_data[:,3], '-', linewidth=2)
ax.plot(Onefil_cnn.reshape(-1, 1), '-', linewidth=2)
plt.yticks(range(0,6,2))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig_2 = plt.figure(figsize=(20, 5))
fig_2.add_subplot(1, 3, 1)
plt.scatter(shuffle_data[:,1], shuffle_data[:,4],  s = 16)
plt.scatter(Onefil_data[:,1], Onefil_data[:,4], color = '#FF8C00', s = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig_2.add_subplot(1, 3, 2)
plt.scatter(shuffle_data[:,0], shuffle_data[:,4], s = 16)
plt.scatter(Onefil_data[:,0], Onefil_data[:,4], color = '#FF8C00',s = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig_2.add_subplot(1, 3, 3)
plt.scatter(shuffle_RHOB, shuffle_GAS, s = 16)
plt.scatter(Onefil_data[:,2], Onefil_data[:,4], color = '#FF8C00', s = 14)
plt.plot(shuffle_RHOB, shuffle_LR_GAS + 1.6, color = 'black')
plt.plot(shuffle_RHOB, shuffle_LR_GAS - 1.6, color = 'black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

