import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
import mzlib as mz


URLname = ""
Filename = ""

#import requests
#r = requests.get(url=URLname)
#data = r.json()

print("Getting data........")
df = pd.read_csv(Filename)
print("data shape = ", df.shape)

Ntop = 25


print("EDA =================================")

print("\nMake ----------------------------")
x = df['Make'].value_counts()  #sorting is automatic with value_counts
list_Make_ALL = list(x.index)
list_Make = list_Make_ALL[:25]
print(list_Make)

print("\nAdding target variable Y---------------------------")
df['Y'] = df['Make'].apply(lambda x: '1' if x in list_Make else '0' )


# Create an equal number of records for Y=0 and Y=1 from Make != null subset.
df_temp = mz.get_balanced_set(df)


print("\nAgency -----------------------------")
df['Agency'].fillna(-1.0,inplace=True)
df['Agency'] = df['Agency'].astype(int)
df['Agency'] = df['Agency'].astype(str)
df['Agency'] = df['Agency'].str.replace("-1", "other")

# Doing this for df_temp -----------
list_Agency = df_temp['Agency'].value_counts().index.tolist()
list_Agency = list_Agency[:10]
list_Agency = [int(c) for c in list_Agency]
list_Agency = [str(c) for c in list_Agency]
df['Agency_new'] = df['Agency'].apply(lambda x: x if x in list_Agency else 'other' )
print(list_Agency)


print("\nColor -----------------------------")
df['Color'] = df['Color'].astype(str)  #just being sure...
list_Color = df_temp['Color'].value_counts().index.tolist()
list_Color = list_Color[:15]
df['Color_new'] = df['Color'].apply(lambda x: x if x in list_Color else 'other' )
print(list_Color)


print("\nViolation code -----------------------------")
df['Violation code'] = df['Violation code'].astype(str)  #just being sure...
list_Code = df_temp['Violation code'].value_counts().index.tolist()
list_Code = list_Code[:15]
df['Code_new'] = df['Violation code'].apply(lambda x: x if x in list_Code else 'other' )
print(list_Code)


# Modeling --------------------------------------
# This set is balances 50/50 for Y=0 and Y=1
df_model = mz.get_balanced_set(df)
df_model.shape

# 80% for training set
df_model_train = df_model.sample(frac=0.8, random_state=1235)
df_model_train.shape

# 20% for testing
df_model_test = df_model.drop( df_model_train.index )
df_model_test.shape

X_train = ['Agency_new', 'Color_new', 'Code_new']
Y_train = ['Y']

dfX = pd.get_dummies(df_model_train[X_train])
dfY = df_model_train[Y_train].values.ravel()

columnsX = dfX.columns.tolist()

rf = RandomForestClassifier( n_estimators=100, oob_score=True, random_state=12345, verbose=2 )
rf.fit( dfX, dfY )

# The OOB score gives a quick estimate of test score
print("OOB score = ", rf.oob_score_)

# Here force everything to be a string, since that is the assumption for the input
agency = "54.0"
color = "GY"
code = "88.13B+"

df_live = mz.get_test_frame(agency, color, code, list_Agency, list_Color, list_Code, columnsX)
predicted = rf.predict( df_live )
mz.print_prediction( predicted[0], Ntop )


print("\nInteractive querying.......")

while(True):
	print("Enter values for: Agency, Color, violation code (press Cntrl-C to exit)")
	val = sys.stdin.readline().split()
	if len(val) != 3:
		print("Enter 3 values.  You entered {}".format(len(val)))
	else:
		agency = val[0]
		color = val[1]
		code = val[2]
		print("You entered: ", agency, color, code)
		df_live = mz.get_test_frame(agency, color, code, list_Agency, list_Color, list_Code, columnsX)
		predicted = rf.predict( df_live )
		mz.print_prediction( predicted[0], Ntop )





