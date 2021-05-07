import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from tqdm import tqdm
import joblib

# defining global variable
feature_path = "D:/Deployment/feature.csv"
label_path = "D:/Deployment/label.csv"
params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'subsample': 0.4,
            'subsample_freq': 1,
            'learning_rate': 0.25,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'lambda_l1': 1,
            'lambda_l2': 1
            }


def read_data(filename):
    # function to read features or labels
    return pd.read_csv(filename)

def train_model(X,Y,folds=2,seed=1,shuffle=True):
    print("-----Divide in folds and start training------")
    kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
    model_errors = []
    models = []

    for train_index, val_index in kf.split(X,Y):
        train_X,val_X = X.iloc[train_index], X.iloc[val_index]
        train_y,val_y = Y['avg_bce'].iloc[val_index],Y.iloc[val_index]
    #     train_y = target.iloc[train_index]
    #     val_y = target.iloc[val_index]
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(val_X, val_y)
        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=(lgb_train, lgb_eval),
                    early_stopping_rounds=100,
                    verbose_eval = 100)
        models.append(gbm)
        predicted_y=gbm.predict(val_X)
        model_error = evaluation(val_y, predicted_y)
        model_errors.append(model_error)
        print(model_errors)
    return models
        

def evaluation(actual, predicted):
    # function for feature_extractor
    message = "function for feature_extractor"
    model_error = mean_squared_error(actual, predicted)  
    # print(model_error)  
    return np.sqrt(model_error)

def save_model(model, modelpath = "../model.pkl"):
	# function for saving_model
	joblib.dump(model, modelpath)
	message = "saved model at... " + modelpath
	print(message)

if __name__=='__main__':
    message = "defining stubs"
    print (message)
    # read data
    features = read_data(feature_path)
    labels = read_data(label_path)
    models=train_model(features, labels)
    model_with_minimum_error = train_model(features, labels)
    save_model(model_with_minimum_error[0])