import os, sys
from pyexpat import features
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import joblib

parkinson_df = pd.read_csv('Data/parkinsons.csv')
pd.set_option('display.max_columns', None)

parkinson_df.head(5)
parkinson_df.describe()
parkinson_df.corr()

#how many suffer from parkinsons vs who dont
sns.countplot(parkinson_df['status'])


fig, ax = plt.subplots(figsize=(12,8))
corr = parkinson_df.corr()
ax = sns.heatmap(corr, vmin= -1, vmax= 1, center= 0, cmap= 
sns.diverging_palette(20,220, n=200))

#Rearrange the columns
parkinson_df = parkinson_df[["name", "MDVP:Fo(Hz)", 
"MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
"MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
"MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", 
"Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", 
"RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]]


#Create a copy of the original dataset
df2= parkinson_df.copy()

#Assign numeric values to the binary and categorical columns
number= LabelEncoder()
df2['name']= number.fit_transform(df2['name'])

#all features & target (status)
X = df2.iloc[:,0:11]
Y = df2.iloc[:,-1]

#selects top 3 features 
best_features = SelectKBest(score_func= chi2, k= 3)
fit = best_features.fit(X, Y)

#creates data frames for scores and features of each score
Parkinson_scores = pd.DataFrame(fit.scores_)
Parkinson_columns = pd.DataFrame(X.columns)

#combines all features and corresponding scores
features_scores = pd.concat([Parkinson_scores, Parkinson_scores],
axis= 1)
features_scores.columns = ['Features', 'Score']
features_scores.sort_values(by= 'Score')

df2.head(5)

#build model
x= parkinson_df[["MDVP:Flo(Hz)", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]]
y= parkinson_df[["status"]]
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size= 0.2, random_state= 7)

model= XGBClassifier()
model.fit(x_train,y_train)

#evaluate model 
y_pred= model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

#defining metrics
y_pred_proba = model.predict_proba(x_test)[::,1]

#Calculate true positive and false positive rates
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)

#Calculate the area under curve to see the model performance
auc= metrics.roc_auc_score(y_test, y_pred_proba)

#Create ROC curve
plt.plot(false_positive_rate, true_positive_rate,label="AUC="+str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4)

plt.show()

# Save the trained model to a file to be used in future predictions 
joblib.dump(model, 'XG.pkl')