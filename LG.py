import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
# To load dataset
df = pd.read_csv("salary1.csv")
df.head()

# to check null values
df.isnull().sum()
# To remove unwanted columns permanently from given dataframe
df.drop("Unnamed: 0", axis=1, inplace=True)
# to check duplicates rows
df.duplicated().sum()

# separate object type data and numeric type data from given dataframe df and hold object type
# data in new dataframe df_obj   and hold numeric type data in new dataframe df_num
# use select_dtypes()  inbuilt method
df_obj = df.select_dtypes('object')
df_num = df.select_dtypes('int64')
# df_num=df.select_dtypes(['int64','float64','int32'])
df['Workclass'].unique()
df['marital-status'].unique()
column = df_obj.columns
# Apply LabelEncoder for object type data

for col in column:
    # create object of LabelEncoder class
    le = LabelEncoder()
    df_obj[col] = le.fit_transform(df_obj[col])

# merge df_num and df_obj and hold new dataframe df_new
df_new = pd.concat([df_obj, df_num], axis=1)
# to remove education-num  column permanently from given dataframe df_new
df_new.drop("education-num", axis=1, inplace=True)

df_new['Income'].value_counts()

# clearly understand , data is unbalanced
# separate input and output from dataframe df_new
X = df_new.drop("Income", axis=1)  # select input
Y = df_new['Income']  # output
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# create object of StandardScaler class
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

Y_train.value_counts()
# create the object of RandomOverSampler class
ros = RandomOverSampler(random_state=1)

# apply RandomOverSampler on training input and output
X_train1, Y_train1 = ros.fit_resample(X_train, Y_train)

# fit_resample()inbuilt method of RandomOverSampler class

# Y_train1.value_counts()

X_test1, Y_test1 = ros.fit_resample(X_test, Y_test)


def create_model(model):  # here create_model() user defined function name
    # model is a user defined parameter which hold the object of algorithm
    model.fit(X_train1, Y_train1)  # train the model with 70% training data
    Y_pred = model.predict(X_test)  # test the model with 30% input
    # print confusion matrix
    print(confusion_matrix(Y_test, Y_pred))
    # print classification report
    print(classification_report(Y_test, Y_pred))
    return model


lr = LogisticRegression()
model = create_model(lr)
