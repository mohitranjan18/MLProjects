import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 

# defining global variable
dataset_path = "C:/Users/Mohit Ranjan/Desktop/Yugen/BlueOptima/pa_work_sample_training_data.csv"
drop_cl=['id_task_session','Remarks on Score','made_submission','js_comments']

data_=pd.read_csv(dataset_path)
dummy_l=['tx_difficulty','total_compute','tx_file_type']
def data_selection(data_,drop_columns=[]):
    '''will rename columns and drop some of the feature'''
    data_.drop(columns=drop_columns,inplace=True,axis=1)
    print(data_.head())
    return data_

def data_validation(data_,dummy_list=[]):
    '''Here we will create dummies and label encode of categorical 
    columns also convert the data type columns if required '''
    if(dummy_list==None):
        dummy_list=dummy_l
    postfix=[]
    # print(dummy_list)
    for values in dummy_list:
        values=values+'_'
        postfix.append(values)
    # print(type(data))
    # print(data_.head())
    data=pd.get_dummies(data=data_,columns=dummy_list,drop_first=True,prefix=postfix)
    # print(data.columns)
    data_=data
    # print(data.head())
    # print(postfix)
    return data_

def data_preprocessing(data_,change_colDtype={}):
    # function for data pre-processing
    '''change_colDtype is a dictionary of columns as keys
     values will be the changed data type'''
    print('------Preprocessing Started---------')
    # print(change_colDtype.keys())
    print(data_.columns)
    for feature in change_colDtype.keys():
        if "bool" in str(data_[feature].dtype):
            data_[feature]= data_[feature].map({True:0,False:1})
        elif 'int' in  str(data_[feature].dtype) or 'float' in str(data[feature].dtype):
            data_[feature]= data_f[feature].astype(change_colDtype[feature])
        else:
            data_[feature]=data_[feature].astype(change_colDtype[feature])
    # print(data_['compile_success'].dtype)
    # print(data_.head())
    return data_
    print('------Preprocessing Ended---------')
def feature_extractor(data_):
    '''
    Deriving new feature
    1.data['nu_pgmr_total'] derving new column as this will take sum of the features 
    such as 0.5*data['nu_pgmr_comment_flux_all']+data['nu_pgmr_cyclo_flux_all']+data['nu_pgmr_filesize_flux_all']+0.5*data['nu_pgmr_dac_flux_all']+data['nu_pgmr_fanout_flux_all']
    +data['nu_loc_flux_source_all']+data['nu_loc_added_source_all']+2*data['nu_ce_models_units_all']
    some of th feature are multiplied by 0.5 ,as they are not that important as it doesn't determine the quality code
    some of the feature which are important are multiplied by 2 
    '''
    data_['nu_pgmr_total']=0.5*data_['nu_pgmr_comment_flux_all']+data_['nu_pgmr_cyclo_flux_all']+data_['nu_pgmr_filesize_flux_all']+0.5*data_['nu_pgmr_dac_flux_all']+data_['nu_pgmr_fanout_flux_all']+data_['nu_loc_flux_source_all']+data_['nu_loc_added_source_all']+2*data_['nu_ce_models_units_all']
    data_['nu_pgmr_total']=data_['nu_pgmr_total']-data_['nu_aberrant_ce_units_all']
    return data_
    # print(data_.columns)

def data_split(data,target):
    # function for splitting data into training and testing
    label=data_[[target]]
    data=data_.drop(target,axis=1)
    return data_,label

if __name__ == '__main__':
    message = "defining stubs"
    data_selection(drop_cl)
    data_=data_validation(data_,dummy_list)
    # print(data_.head())
    data_preprocessing({'compile_success':'int64'})
    feature_extractor()
    train_,label=data_split(data_,'avg_bce')
    data_.to_csv("../feature.csv", index=False)
    label.to_csv("../label.csv", index=False)
    print ('selected data')
   