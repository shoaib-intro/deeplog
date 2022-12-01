# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:54:24 2022

@author: ashoaib
"""

import pandas as pd
import numpy as np
from math import sqrt
import re, json, os, joblib
import scipy.stats as st
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
'''

def __init__(self, options):
    self.device = options['device']
    self.input_dir = options['input_dir']
    self.output_dir = options['output_dir']
    self.log_file = options['log_file']
    
    # model
    self.window_size = options['window_size']
    self.num_candidate = self.window_size - 1
    self.num_epochs = options['num_epochs']
    self.batch_size = options['batch_size']
    self.input_size = options['input_size']
    self.num_layers = 2
    self.hidden_size = 64
    
    with open(self.output_dir + "log_keys.json", "r") as f:
        event_num = json.load(f)
    self.num_classes = len(event_num) +1
    
    model = Model(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)


'''
df = pd.read_pickle('output/efs_logs.txt_structured.pkl')


def cleaning(df):
    # datetime cleaning
    #df['DateTime'] =df['DateTime'].apply(lambda x: re.sub('[A-Z]', ' ',x))
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S') # formate according to time zone

    df['Time_diff'] = pd.to_datetime(df['DateTime'].astype(str)).diff().dt.total_seconds().fillna(0) # difference  from previous line
    
    df['Time_diff'] = df['Time_diff'].clip(lower=0) # -> if any time differ is less than zero

    # cleaning
    df['cleaned'] = df['ParameterList'].apply(lambda x: re.sub('[\[|\]|\'|\"|\|\s+|\.|\-]', '', str(x)).split(','))

    # tokenizing

    myTokenizer = Tokenizer()
    myTokenizer.fit_on_texts(df['cleaned'])
    df['embeded']= myTokenizer.texts_to_sequences(df['cleaned'])
    
    # insert time difference at first
    for time, padded in zip(df['Time_diff'], df['embeded']):
        padded.insert(0, int(time))


    # change logkeys into int
    with open("output/log_keys.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))
    unique_keys = df['EventId'].unique()
    
    # save matrix of each logkey
    if not os.path.isdir('output/temp/matrix/'): os.makedirs('output/temp/matrix/')
    filename = 'output/temp/matrix/'
    for key in unique_keys:
        data = df[df['EventId']==key]
        list_array = data['embeded']
        #print(list_array)
        
        eventID = str(key)
        if os.path.exists(filename):
            np.save(filename + eventID+'.npy', list_array)
        else:
            os.mkdir(filename)
            np.save(filename + eventID+'.npy', list_array)
    return df

# cleaning
df = cleaning(df)


# input output sequences
def training_data_generate(matrix, n_steps):
    '''
    :param matrix: the paramter value vectors for a single log key
    :param n_steps_in: the length of sequence, which depends on how long the matrix is
    :param n_steps_out: always one, we just need one really parameter vector
    :return:
    '''
    X, Y = list(), list()
    for i in range(matrix.shape[0]):
        # find the end of this pattern
        end_ix = i+n_steps
        # check whether beyond the dataset
        if end_ix > matrix.shape[0]-1:
            break
        seq_x, seq_y = matrix[i:end_ix, :], matrix[end_ix, :]
        X.append(seq_x)
        Y.append(seq_y)
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype= np.float32)
    print("the shape of X is:",X.shape)
    
    return X, Y


def mean_squared_error_modified(y_true, y_pred):
    d_matrix = np.subtract(y_true, y_pred)
    print("the d_matrix is:", d_matrix)
    means = []
    for i in range(d_matrix.shape[1]):
        means.append(np.mean(d_matrix[:, i] * d_matrix[:, i], axis=-1))
    print("the means are:", means)
    return np.mean(means), means

def confidence_interval(confidence, mse):
    ''' function to compute the confidence interval boundaries
    :param confidence: the confidence value or threshold, like 98%
    :param mses_list: the errors list
    :return: the boundaries
    '''
    # define the interval tuple
    return st.t.interval(confidence, len(mse)-1, loc=np.mean(mse), scale=st.sem(mse))

def anomaly_report(mses_list,file_number):
    # here we use the max value as the threshold
    confidence_intervial_fp1 = confidence_interval(0.98, mses_list)
    # it is for the false positive detection
    threshold1 = confidence_intervial_fp1[1]
    confidence_intervial_fp2 = confidence_interval(0.99, mses_list)
    # it is for the false positive detection
    threshold2 = confidence_intervial_fp2[1]
    confidence_intervial_an = confidence_interval(0.999, mses_list)
    # it is for the anomaly detection
    threshold3 = confidence_intervial_an[1]
    # record the potential anomaly logs
    suspicious_logs = []
    # record the false positive logs
    fp_logs = []
    for i in range(len(mses_list)):
        if mses_list[i] > threshold3:
            print('The {}th log in matrix {} is suspiciously anomaly'.format(i, file_number[0]))
            suspicious_logs.append(i)
        elif mses_list[i] > threshold1:
            print('The {}th log in matrix {} is false positive'.format(i, file_number[0]))
            fp_logs.append(i)
        else:
            continue
    return threshold1, threshold2, threshold3, suspicious_logs, fp_logs

from keras import Sequential
from keras.layers import Dense, LSTM
def LSTM_model(trainx, trainy):
    # use the train
    model = Sequential()
    model.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape=(trainx.shape[1], trainx.shape[2])))
    model.add(LSTM(100, activation = 'relu'))
    model.add(Dense(trainx.shape[2]))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(trainx, trainy, epochs = 50, verbose=2, callbacks=[callbacks])
    model.fit(trainx, trainy, epochs=50, verbose=2)
    model.summary()
    joblib.dump(model,'model.pkl')
    return model



# load matrix and if there is rmse
filenames = []
root_dir = 'output/temp/matrix/'
# r=root, d = directories, f=files
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
else:
    for r, d, f in os.walk(root_dir):
        for file in f:
            if file.endswith('.npy'):
                filenames.append(os.path.join(r, file))
# set the random seed
seed = 0
rmses = []
rmses_dict = {}
    
# record the anomaly logs with the name of file and the anomaly logs order
suspicious_anomaly_dict, fp_logs_dict = {}, {}
len(filenames)

for file in filenames:
    if os.path.isfile(file + '_rmses.pkl'):
        rmses = joblib.load(file + '_rmses.pkl')
    else:
        # looping read single file
        print("we are processing matrix:", file)
        matrix = np.load(file, allow_pickle=True)
        #matrix = np.array(matrix).reshape(matrix.shape[0],-1)
        # set n_steps_in and n_steps_out depending on the sequence length of matrix
        # we set the test_size=0.4, the length of matrix should be at least 8
        # Here, I will change the length of history to see the performance

        if matrix.shape[0] >= 8:
            n_steps = 3
            X, Y = training_data_generate(matrix, n_steps)
            train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.4, random_state=seed)
            # reshape x to (samples, time steps, features)
            #train_X = np.array(X).reshape(-1, n_steps, len(data[0]))
            # reshape y to (samples, features)
            #train_y = np.array(y).reshape(-1, len(data[0]))
        else:
            continue
    
        # get the model
        # reshape x to (samples, time steps, features)
        train_x = np.array(X).reshape(-1, n_steps, len(train_x[0]))
        # reshape y to (samples, features)
        train_y = np.array().reshape(-1, len(train_y[0]))
        #print(f'trainx {train_x.shape[1]},    trainx 2 : {train_x.shape[2]} \t {train_x}')
        model = LSTM_model(train_x, train_y)
        print("the test_x is:", test_x)
        # make a prediction
        yhat = model.predict(test_x)
        # delete the time step element
        print("the predicted y is:", yhat)
        
        rmse, means = mean_squared_error_modified(test_y, yhat)
        # rmse, meams = mean_squared_error(test_y, yhat)
        rmse = sqrt(rmse)
        print('Test RMSE: %.3f' % rmse)
        # use the mean square error to compare the difference between predicted y and validation y
        # the error follows the Gaussian distribution ---- normal, otherwise abnormal
        rmses.append(rmse)
        # save the result
        rmses_dict[file] = rmse
        
        # save the results to files
        joblib.dump(rmses, file + '_rmses.pkl')
        joblib.dump(rmses_dict, file + '_rmses_dict.pkl')
    
    
    # ===== part to predict the anomaly logs ====
    file_number = re.findall('\d+', file)
    threshold1, threshold2, threshold3, suspicious_logs, fp_logs = anomaly_report(means,file_number)
 
    # part to print the picture of means with bar chart
    # create the x axis labels for plot
    x_list = []
    for i in range(len(rmses)):
        x_list.append(i)
    if len(x_list)<=1:
        pass
    else:
        # part to print the picture of means with line chart
        if len(suspicious_logs)==0:
            plt.plot(x_list, rmses)
        else:
            plt.plot(x_list, rmses, 'r')
        # add the threshold lines with percentage
        print(threshold1,threshold2,threshold3)
        plt.axhline(y=threshold1, linestyle = "-", label = '98%', color='blue')
        plt.axhline(y=threshold2, linestyle = "-.", label = '99%', color='green')
        plt.axhline(y=threshold3, linestyle = "--", label = '99.9%', color='orange')
        plt.ylabel("Errors Values")

        plt.title(file_number[0] + ' ' + 'Errors Distribution')
        plt.title(file + ' ' + 'Errors Distribution')
        plt.show()
    # generate the dict about anomaly and false positive logs
    if len(suspicious_logs) == 0 & len(fp_logs) == 0:
        pass
    else:
        suspicious_anomaly_dict[file_number[0]] = suspicious_logs
        fp_logs_dict[file_number[0]] = fp_logs
# save the result
joblib.dump(suspicious_anomaly_dict,'output/temp/suspecious_logs_param.csv')
joblib.dump(fp_logs_dict, 'output/temp/fp_logs.csv')



def LSTM_model(trainx, trainy):
    # use the train
    model = Sequential()
    input_shape=(trainx.shape[1], trainx.shape[2])
    model.add(LSTM(100, activation = 'relu', return_sequences = True,input_shape = input_shape))
    print(input_shape)
    model.add(LSTM(100, activation = 'relu'))
    model.add(Dense(trainx.shape[2]))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(trainx, trainy, epochs = 50, verbose=2, callbacks=[callbacks])
    model.fit(trainx, trainy, epochs=50, verbose=2)
    model.summary()
    joblib.dump(model,'model.pkl')
    return model




test = np.load('output/temp/matrix/63.npy', allow_pickle=True)
test = np.array(test).reshape(test.shape[0],-1)

x, y = training_data_generate(test, 3)
# reshape x to (samples, time steps, features)
train_X = np.array(test).reshape(-1, 3, len(x[0]))
# reshape y to (samples, features)
train_y = np.array(y).reshape(-1, len(x[0]))
model = LSTM_model(train_X, train_y)


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=seed)
train_x = np.asarray(train_x).astype('float32')
train_y = np.asarray(train_y)
model = LSTM_model(train_X, train_y)


train_x.shape[2]



X, Y = training_data_generate(test, 3)
# reshape x to (samples, time steps, features)
test = np.array().reshape(-1, len(test[0]))
# reshape y to (samples, features)
train_y = np.array(y).reshape(-1, len(data[0]))