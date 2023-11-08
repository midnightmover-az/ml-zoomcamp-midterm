import pandas as pd
import numpy as np

from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Ridge

from IPython.display import display

print( 'Pandas version = ' + pd.__version__ + '\n' )

import seaborn as sns
#from matplotlib import pyplot as plt
#%matplotlib inline

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))



################################################################################
######  Read the data from the file  ###########################################
################################################################################

df = pd.read_csv('heart_attack_prediction_dataset.csv')
df.head()


################################################################################
######  Column Name  Cleanup ###################################################
################################################################################

df.columns = df.columns.str.replace(' ', '_').str.lower()
booleanColumns = [
    'diabetes',
    'family_history',
    'smoking',
    'obesity',
    'alcohol_consumption',
    'previous_heart_problems',
    'medication_use',
    'heart_attack_risk'
]
categoricalColums = [ 'sex', 'country', 'continent', 'hemisphere' ]


################################################################################
######  Data Cleanup ###########################################################
################################################################################

# Patient_id feature has no relevance for hart atack risk so drop it
# These two comlims 'continent' and 'hemisphere' are depemdent on country so delete them
# Obesity and BMI are redundant
# physical_activity_days_per_week is redundant with "exercise_hours_per_week". Removing physical_activity_days_per_week

ignoreDiastolicPressure = True

ignoredFeatures = [  'patient_id', 'continent', 'hemisphere', 'country', 'income', 'obesity', 
                'physical_activity_days_per_week', 'sedentary_hours_per_day', 'exercise_hours_per_week', 'previous_heart_problems',
                'diet', 'stress_level', 'sleep_hours_per_day', 'alcohol_consumption', 'diabetes', 'sex', 'medication_use', 'bmi',
                ]
for col in ignoredFeatures:
    del df[ col ]

########################################################################################################################
######  Transform 'sex' feature from 'Male' and 'Female' to 0 and 1 ####################################################
########################################################################################################################

if not 'sex' in ignoredFeatures:
    df['sex'] = df['sex'].astype("string")
    df['sex'] =  df['sex'].apply({'Male':1, 'Female':0}.get)
else:
    print( "Ignoring sex column" )
    print(df.columns)
df.head


########################################################################################################################
######  Split  blood_pressure into two new featuresnumerical 'blood_pressure_sistolic','blood_pressure_diastolic' ############
########################################################################################################################

df[['blood_pressure_sistolic','blood_pressure_diastolic']] = df.blood_pressure.str.split("/",expand=True) 
#print( df[ ['blood_pressure', 'blood_pressure_sistolic', 'blood_pressure_diastolic'] ] )

# data from 'blood_pressure' is now contained in 'blood_pressure_sistolic' and 'blood_pressure_diastolic' so it is redundant and must be deleted
del df[ 'blood_pressure' ]

df['blood_pressure_sistolic'] = pd.to_numeric(df['blood_pressure_sistolic'])
if not ignoreDiastolicPressure:
    df['blood_pressure_diastolic'] = pd.to_numeric(df['blood_pressure_diastolic'])
else:
    del  df['blood_pressure_diastolic']
    print( "Ignoring Diastolic Pressure")

print(df.columns)
#pd.set_option('display.max_columns', None)
display(df)
#df.head()


########################################################################################################################
######  Diet feature is caegorical so we have to handle it appropriately  ##############################################
########################################################################################################################

if not 'diet' in ignoredFeatures:

    dfDietDummies = pd.get_dummies(df['diet'], prefix='diet', drop_first='True') 

    # Covert from Boolean to 0 and 1 
    dfDietDummies.replace({False: 0, True: 1}, inplace=True)

    df = pd.concat([df,dfDietDummies],axis=1)

    # now drop the original 'country' column (you don't need it anymore)
    df.drop(['diet'],axis=1, inplace=True)

    df.columns = df.columns.str.lower()
else:
    print( "Ignoring diet column" )
    print(df.columns)
    display( df )


########################################################################################################################
######  Country feature is caegorical so we have to handle it appropriately  ###########################################
########################################################################################################################

if not 'country' in ignoredFeatures:
    dfCountryDummies = pd.get_dummies(df['country'], prefix='country', drop_first='True') 

    # Covert from Boolean to 0 nd 1 
    dfCountryDummies.replace({False: 0, True: 1}, inplace=True)

    df = pd.concat([df,dfCountryDummies],axis=1)

    df.columns = df.columns.str.lower()
    # now drop the original 'country' column (you don't need it anymore)
    df.drop(['country'],axis=1, inplace=True)
else:
    print( "Ignoring country column" )

print(df.columns)
display( df )


########################################################################################################################
######  Split the data in train eval and test. Full train data is train data plus eval data   ##########################
########################################################################################################################

seed = 1

print (df.isna().any())

df_intermediate_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_full_train = df_intermediate_train.copy()
df_train, df_val = train_test_split(df_intermediate_train, test_size=0.25, random_state=seed)

y_train = df_train['heart_attack_risk'].values
y_full_train = df_full_train['heart_attack_risk'].values
y_val = df_val['heart_attack_risk'].values
y_test = df_test['heart_attack_risk'].values

del df_train['heart_attack_risk']
del df_full_train['heart_attack_risk']
del df_val['heart_attack_risk']
del df_test['heart_attack_risk']

df_train.info()

print ("##########################")
display( y_full_train )


########################################################################################################################
######  Train the Model  ###############################################################################################
########################################################################################################################

treshHold = 0.5

from sklearn.linear_model import LogisticRegression
logmodel2 = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=2000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
logmodel2.fit(df_full_train,y_full_train)

predictionsProba = logmodel2.predict_proba(df_test)[:, 1]
#display(predictionsProba)
predictions = (predictionsProba >= treshHold)
#display(predictions)

accuracy = accuracy_score(y_test, predictions)
print( "Logistic Full Train Regression Accuracy=", accuracy )


########################################################################################################################
######  Save the Model to a file  ######################################################################################
########################################################################################################################

import pickle

#output_file = f'regressionModel.bin'
output_file = f'logisticRegressionModel.bin'

# Sccond Logistic Regression model has good enough performance so I am saving that model

with open(output_file, 'wb') as f_out:
    pickle.dump((logmodel2), f_out)

print(f'the model is saved to {output_file}')
