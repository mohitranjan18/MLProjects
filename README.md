


**Requirements**

Python 3.5+
NumPy (pip install numpy)
Pandas (pip install pandas)
Scikit-learn (pip install scikit-learn)
MatplotLib (pip install matplotlib)
Seaborn (pip install seaborn)
lightgbm
Flask (pip install flask)

**Light GBM model for Regression**

1. What is Light GBM?
Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.

Since it is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’.

Leaf wise splits lead to increase in complexity and may lead to overfitting and it can be overcome by specifying another parameter max-depth which specifies the depth to which splitting will occur.

Below, we will see the steps to install Light GBM and run a model using it. We will be comparing the results with XGBOOST results to prove that you should take Light GBM in a ‘LIGHT MANNER’.

Let us look at some of the advantages of Light GBM.

 

2. Advantages of Light GBM
Faster training speed and higher efficiency: Light GBM use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure.
Lower memory usage: Replaces continuous values to discrete bins which result in lower memory usage.
Better accuracy than any other boosting algorithm: It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
Compatibility with Large Datasets: It is capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST.
Parallel learning supported.



**Approaching the Solution**
Step One: Collecting Data:

Data was provided ,no efforts was required to collect the data
**
**Step Two: Exploratory Data Analysis****

Obeservations:

1.No null values were found 
2.Dropped column such as made_submission as it had had no variations ,review comments as the questions did not decide the quality of code written 

**Step Three: Feature Engineering****

1.Created dummies and label encoded the categorical variables
2.Derived new columns 
3.Selected the training data

*****Step Four: Training the Model ****

Trained the model using LightGBM algorithm as there was no signicant linear realation ship between dependent and independent variable
Used Kfold methods for training purpose (kept fold=2) as data point were vere low in number
Measured the model  using MSE and saved  the best model .



 ******Steps to run Notebook *******
 
 1. Set data set path in global variable of feature.py class (Eg: dataset_path = "C:/Users/Mohit Ranjan/Desktop/Yugen/BlueOptima/pa_work_sample_training_data.csv")
 2. Run feature.py 
 3. There will be two file saved at your ../ ,labels and feature.csv
 4. Copy this path and put in train.py global variables
 5. run train.py
 6. Saved the best model
 7. See output for rmse

Note: Please don't run app.py as it is incomplete and but it can be used to write api to fetch model output 


******Below are the questions asked along with the answers *******

You should pay attention to the following considerations when designing your solution :
● Is the model overfitting? The model is not overfitting as there is very less difference between train and val error
● Is the model retrainable given additional data? The model is retrainable with additional data .In case of new feature is added ,it won't take much effort to train the model again also as there was less datapoint so didn't add condition to handle Null values

***Summary**

  training's rmse: 0.776794       valid_1's rmse: 0.732964
  
 *********Productionization of Model************
 
 Efforts will be very less as all the feature engineering  has been done using methods with standard inputs 
 It won't take much effort to add api's in app.py to fetch the model output
 
  
  
 
**** How Could I have improved  model?****
 
 1.Give the more data points ,I could have done more data analysis in order to understand the data
 2.I could have tried SVM regressor or other Advance regression techniques
 3.Could Have done more feature engineering and dropped some redundant and similar variable
 








