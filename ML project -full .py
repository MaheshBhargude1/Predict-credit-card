#!/usr/bin/env python
# coding: utf-8

# # Predicting whether a credit card  will be approved or rejected

#             Our Proposal is important in Todays world because it addresses the need for acuurate client prediction and its potential impact on banking sector.Predicting a good client is crucial for bank as it enables them to make informed decisions about granting loan,offering credit and managing the risk.
#             In the context of india,where the proposed method is relevent,there may be knowledge gap in terms of comprehensive credit assessment for clients,particulary for individual or bussinesses with limited credit history or from underserved segment of population.
#             The proposed method can bridge thise gap by leaverging the advanced dataanalytics tehnique,including machin learning and artificial intelligence,to analyze the alternative data source and extract valuable insights.
#       
#       

# # Initial Hypothesis

# Hypothesis 1: Individuals with higher annual income are more likely to have their credit card approved.
# 

# Hypothesis 2: Married individuals are more likely to have their credit card approved compared to single individuals.
# 

# Hypothesis 3: People with property ownership are more likely to get their credit card approved.
# 

# Hypothesis 4: The number of children a person has does not significantly impact credit card approval.
# 

# Hypothesis 5: Individuals with a stable and longer employment history are more likely to get their credit card approved.
# 

# Hypothesis 6: Educational qualification does not have a substantial impact on credit card approval.
# 

# Hypothesis 7: Individuals with a higher number of family members are more likely to get their credit card approved.
# 

# Hypothesis 8: Gender does not play a significant role in credit card approval decisions.

# In[1]:


# Importing various labraries

import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


# In[3]:


# Import Dataset
creditcard_raw=pd.read_csv(r"C:\Users\ADMIN\Downloads\Dataset (2)\Credit_card.csv")
creditcard_raw.head()


# In[ ]:





# The person whose annual income is more as well as if he owns property then chances of credit approval will be more.. 

# In[4]:


# Create a copy
df1=creditcard_raw.copy()
df1.head(1)


# In[5]:


# Import another dataset
credit_label=pd.read_csv(r"C:\Users\ADMIN\Downloads\Dataset (3)\Credit_card_label.csv")
credit_label.head(1)


# In[6]:


# Create a copy
df2=credit_label.copy()
df2.head(1)


# In[7]:


# Mearge Both dataset
result=pd.merge(df1,df2,on="Ind_ID",how="left")
result.head(1)


# In[42]:


# Check the shape
result.shape


# Here,There are 1548 rows and 19 columns

# In[74]:


# Check first five rows
result.head()


# We check First Five observation of table along with that check whether there is any Null values or not.In Type_occupation Column there are so many missing values.

# In[75]:


# check last five rows
result.tail()


# We check Last Five observation of table along with that check whether there is any Null values or not.

# In[76]:


# check all columns
result.columns


# Here column Names seem correct.

# In[79]:


# Detail regarding the dataset in the below command
result.info()


# In[80]:


# nuniuqe function to count unique values in each column
result.nunique()


# In[104]:


# Statistics of all coilumn
result.describe(include= 'all').T


# In[81]:


# Check the datatypes  
result.dtypes


# In[61]:


result.var()


# The data Points in Annual_income,Birthday_count,Employed_days are highly spread out from the mean.

# In[62]:


result.std()


# The data points in Employed_days are highly dispersed from the mean.

# In[9]:


# Count of all null values in columns
result.isnull().sum()


# In type of occuaption there are lot of missing values.

# In[10]:


# check if there are duplicates
result.drop_duplicates()


# There are No any duplicates.

# # Analyze the each Column data

# In[85]:


result["GENDER"].unique()


# In[86]:


result["Car_Owner"].unique()


# In[87]:


result["Propert_Owner"].unique()


# In[88]:


result["CHILDREN"].unique()


# In[89]:


result["Annual_income"].unique()


# In[90]:


result["Type_Income"].unique()


# In[91]:


result["EDUCATION"].unique()


# In[92]:


result["Marital_status"].unique()


# In[93]:


result["Housing_type"].unique()


# In[94]:


result["Birthday_count"].unique()


# In[95]:


result["Employed_days"].unique()


# In[54]:


result["Mobile_phone"].unique()


# In[55]:


result["Work_Phone"].unique()


# In[98]:


result["Phone"].unique()


# In[99]:


result["EMAIL_ID"].unique()


# In[100]:


result["Type_Occupation"].unique()


# In[101]:


result["Family_Members"].unique()


# In[102]:


result["label"].unique()


# # Explore the data
# 

# In[ ]:





# In[8]:


# create histogram
plt.figure(figsize=(4,3))
fig = px.histogram(result, x="GENDER", nbins=15,  height = 500, width = 500,title="GENDER")
fig.show()
print(result.GENDER.value_counts())


# In[ ]:





#  In the above graph Count of Female is greater than male

# In[9]:


# create histogram
plt.figure(figsize=(4,3))
fig = px.histogram(result, x="Car_Owner", nbins=15,  height = 500, width = 500,title="Car_Owner")
fig.show()
print(result.Car_Owner.value_counts())


# Most of the people do not have car.

# In[67]:


# create countplot
plt.figure(figsize=(4,3))
sns.countplot(data=result,x="Propert_Owner")
plt.title("Property Owner")
plt.show()
print(result.Propert_Owner.value_counts())


# Most of the people have property and some people dont have property.

# In[64]:


plt.figure(figsize=(4,3))
fig = px.histogram(result, x="Propert_Owner",color="GENDER",nbins=15,  height = 500, width = 500)
fig.show()


# Here,Most of the Property owners are Female.There count is greater than male.

# In[11]:


# create countplot
plt.figure(figsize=(4,3))
sns.countplot(data=result,x="CHILDREN")
plt.show()
print(result.CHILDREN.value_counts())


# Most people do not have children.There are some people who have 14 children.  .

# In[12]:


# create histplot
plt.figure(figsize=(4,3))
sns.histplot(result,x="Annual_income",kde=True,bins=20)
plt.show()


# Most of people have annual income from 1 lac to 2 lac.Thise is right skewed data.

# In[63]:


# create histogram
plt.figure(figsize=(4,3))
fig = px.histogram(result, x="EDUCATION",color="GENDER",nbins=15,  height = 500, width = 500)
fig.show()


#   Most People have taken education upto Secondary/secondary special.In thise count of female is greater than male.
#    There are Fewer people with degrees.

# In[137]:


# create countplot
plt.figure(figsize=(4,3))
sns.countplot(data=result,y="Marital_status")
plt.show()


# Count of Number of married people is More .Widow are less.

# In[14]:


# creat histogram
plt.figure(figsize=(4,3))
fig = px.histogram(result, x="Type_Occupation", nbins=15,  height = 500, width = 500)
fig.show()


# Most of people are working as Laborer.

# In[69]:


# creat histogram
plt.figure(figsize=(4,3))
fig = px.histogram(result, x="Employed_days", nbins=15,  height = 500, width = 500)
fig.show()


# In[16]:


#create count plot 
plt.figure(figsize=(4,3))
sns.countplot(data=result,x="Family_Members",hue="GENDER")
plt.show()


# The Number of people having 2 people in the house is more.
# The Number of people having 15 people in the house is less.

# In[145]:


# create countplot
plt.figure(figsize=(4,3))
sns.countplot(data=result,x="label",color="g")
plt.show()


# In the above graph,The number of people who have been approved for credit cards is high.

# In[17]:


# create histogram
plt.figure(figsize=(4,3))
fig = px.histogram(result, x="Housing_type", nbins=15,  height = 500, width = 500)
fig.show()


# Count of house apartment is more than other housing type.

# In[147]:


#create countplot
plt.figure(figsize=(4,3))
sns.countplot(data=result,x="Work_Phone")
plt.show()


# Most people dont have a work phone.

# In[148]:


# create countplot
plt.figure(figsize=(4,3))
sns.countplot(data=result,x="Phone")
plt.show()


# In[154]:


# creat bar plot
plt.figure(figsize=(4,3))
sns.barplot(x ='Type_Income', y ='Annual_income', data = result,width=0.8)


# In[70]:


plt.figure(figsize=(4,3))

fig = px.histogram(result, x="Type_Income",y ='Annual_income' ,nbins=15,  height = 500, width = 500)
fig.show()


# The annual income of Working People is more than other profession.

# In[157]:


sns.pairplot(result)


# # MIssing Value

# In[9]:


# import missingno library
import missingno as msno


# In[10]:


# creat matrix
msno.matrix(result)


# There are lot of missing values in Type_occupation.some missing values in Birthday_count ,Gender column.

# In[12]:


sorted=result.sort_values("Type_Occupation")
msno.matrix(sorted)


# There is no any relation between missing values of Type_occupation column and others columns.
#           so we can say that they are Missing Complately at Random (MCAR).

# In[13]:


# creat heatmap
msno.heatmap(result)


# There is no any corelation between the columns.

# In[14]:


# correlation between two variable
result.corr()


# In[15]:


import plotly.figure_factory as ff
result_corr=result.corr()
x=list(result_corr.columns)
y=list(result_corr.index)
z=np.array(result_corr)
fig=ff.create_annotated_heatmap(x=x,y=y,z=z,annotation_text=np.round(z,2),hoverinfo="z")
fig.show()


# There is Positive correlation between children and Family member.

# # Outlier Detection

# In[43]:


result.skew()


# The Children  and annual income columns are highly positively skew.

# # Outlier Detection for Family_Member Column

# In[53]:


# for family member column
sns.boxplot(y ='Family_Members', data = result)


# Here,we can see there is a outlier.

# In[16]:


Q1 = result['Family_Members'].quantile(0.25)
Q3 = result['Family_Members'].quantile(0.75)
print("Q1==", Q1)
print("Q3==",Q3)
IQR = Q3 - Q1
print("IQR",IQR)
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
print("upper_bound",upper_bound)
print("lower_bound",lower_bound)


# In[18]:


result = result[result.Family_Members <= upper_bound]
result = result[result.Family_Members  >= lower_bound]


# # After Trimming in Family_Member Column

# In[19]:


sns.boxplot(y ='Family_Members', data = result)


# # Outlier Detection for Children coulmn

# In[44]:


# for children column
sns.boxplot(y ='CHILDREN', data = result)


# Here,we can clearly see that there are some outlier in Children column. 

# In[45]:


Q1 = result['CHILDREN'].quantile(0.25)
Q3 = result['CHILDREN'].quantile(0.75)
print("Q1==", Q1)
print("Q3==",Q3)
IQR = Q3 - Q1
print("IQR",IQR)
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
print("upper_bound",upper_bound)
print("lower_bound",lower_bound)


# In[46]:


result = result[result.CHILDREN <= upper_bound]
result = result[result.CHILDREN  >= lower_bound]


# # After Trimming in Children column

# In[47]:


sns.boxplot(y ='CHILDREN', data = result)


# All outliers in children column are removed.

# # Outlier Detection in Annual_Income Column

# In[59]:


# create hist plot for Annual_income
sns.histplot(data = result['Annual_income'])
plt.axvline(x=result.Annual_income.mean(),color='red',alpha=0.5,label='Mean')
plt.axvline(x=result.Annual_income.median(),c='blue',ls='--',alpha=0.5,label='Median')
plt.legend()


# here,In graph,Mean is greater than median.So we can say that thise is right skewed data.

# In[48]:


# create box plot for Annual_income
sns.boxplot(y ='Annual_income', data = result)


# In[50]:


# for calculation of Q1,Q2,and IQR
Q1 = result['Annual_income'].quantile(0.25)
Q3 = result['Annual_income'].quantile(0.75)
print("Q3==",Q3)
print("Q1==", Q1)
IQR = Q3 - Q1
print("IQR",IQR)
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
print("upper_bound",upper_bound)
print("lower_bound",lower_bound)


# In[74]:


result = result[result.Annual_income <= upper_bound]
result = result[result.Annual_income  >= lower_bound]


# Anything beyond upper bound and below lower bound are considered as outlier and they will be removed.

# # After trimming In Annual_Income Column

# In[75]:


sns.boxplot(y ='Annual_income', data = result)


# Now we have removed the outliers from Annual income column.

# In[169]:


# create a copy
result12=result.copy()
result12.head()


# # Convert categorical to numerical data by using map method

# In[170]:


# for car owner
result12['Car_Owner'] = result12['Car_Owner'].map({'Y':0,'N':1})
result12.head()

# for propert Owner
result12['Propert_Owner'] = result12['Propert_Owner'].map({'Y':0,'N':1})
result12.head()

# for gender
result12['GENDER'] = result12['GENDER'].map({'M':0,'F':1})
result12.head()


# # Encoding

# In[171]:


from sklearn.preprocessing import LabelEncoder
for i in ["Type_Income","EDUCATION","Marital_status","Housing_type","Housing_type","Type_Occupation"]:
    result12[i]=LabelEncoder().fit_transform(result12[i])
result12.head()


# # Imputation Techniques

# ---SimpleImputer

# In[172]:


from sklearn.impute import SimpleImputer 
result12_median = result12.copy() 
median_imputer = SimpleImputer(strategy='median')   # apply median strategy
result12_median.iloc[:, :] = median_imputer.fit_transform(result12_median)


# --KNN Imputer

# In[173]:


#import KNN 
from fancyimpute import KNN 
knn_imputer = KNN() 
result12_knn = result12.copy(deep=True) 
result12_knn.iloc[:, :] = knn_imputer.fit_transform(result12_knn)


# -- Mice Imputer

# In[174]:


# import IterativeImputer
from fancyimpute import IterativeImputer      
MICE_imputer = IterativeImputer()             
result12_MICE = result12.copy(deep=True)      
result12_MICE.iloc[:, :] = MICE_imputer.fit_transform(result12_MICE)


# #  Visualizing the imputation technique

# In[175]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 12)) # creating 2 rows and 3 columns
nullity = result12['Birthday_count'].isnull() + result12['Annual_income'].isnull() # creating null columns between culmen length and culmen depth
imputations = {                                       # creating a python dictionary
               'Median Imputation': result12_median,
               'KNN Imputation': result12_knn,
              'MICE Imputation': result12_MICE}

for ax, df_key in zip(axes.flatten(), imputations):    # a for loop to iterate over the subplots and the imputed data
    imputations[df_key].plot(x='Birthday_count', y='Annual_income', kind='scatter',
                             alpha=0.5, c=nullity, cmap='rainbow', ax=ax,
                             colorbar=False, title=df_key)


# Here,KNN imputation seems good as compared to others.So we can select the knn Imputation.

# In[177]:


result12_knn.isnull().sum()


# All the null values are now filled.

# # Separating Independent and Dependent Variable

# In[89]:


X=result12_knn.iloc[:,:-1]
y=result12_knn.iloc[:,-1]


# In[90]:


X.head(1)


# In[91]:


y.head(1)


# # Feature selection

# In[92]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result12=logit_model.fit()
print(result12.summary2())


# From the above ,Ind_ID,Car_Owner,Propert_Owner,Annual_income,EDUCATION,Housing_type,Birthday_count,Employed_days,Mobile_phone,
# Work_Phone,Phone,EMAIL_ID,Type_Occupation are thise variable are not good predictors.So we can remove it.

# In[93]:


#Drop the unnecessary variable
X=X.drop(["Ind_ID" ,"Car_Owner","Propert_Owner","Annual_income","EDUCATION","Housing_type","Birthday_count","Employed_days","Mobile_phone","Work_Phone","Phone","EMAIL_ID","Type_Occupation"],axis=1)
X


# In[94]:


y


# # Spliting data into X train and y train

# In[95]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[96]:


print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# # Feature Scaling

# -To remove the extra influence of high numerical value variable

# In[97]:


# performing standardization technique
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
# apply to X_train
X_train=sc.fit_transform(X_train)
X_train


# In[98]:


# apply to X_test
X_test=sc.fit_transform(X_test)
X_test


# # Training Various Models

# Our Dependent variable is BINARY.So we can use here CLASSIFIER technique.
# In thise we are performing the 1.LogisticRegression 2.SVC 3.GradientBoostingClassifier 4.RandomForestClassifier  techniques.

# In[99]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[100]:


# logistic regresssion
from sklearn.linear_model import LogisticRegression
Lr=LogisticRegression()

#Support vector regression
from sklearn.svm import SVC
sv=SVC()

#XG boost
from sklearn.ensemble import GradientBoostingClassifier
xg=GradientBoostingClassifier()

#random Forest
from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier()



# In[101]:


Lr.fit(X_train,y_train)


# In[102]:


sv.fit(X_train,y_train)


# In[103]:


xg.fit(X_train,y_train)


# In[104]:


rc.fit(X_train,y_train)


# In[106]:


predict1=Lr.predict(X_test)  
predict2=sv.predict(X_test)
predict3=xg.predict(X_test)
predict4=rc.predict(X_test)


# In[107]:


# Compare the four Models
df=pd.DataFrame({"Actual_Value":y_test,"Logistic_R":predict1,"SVR":predict2,"Gradient_Boost":predict3,"Random_Forest":predict4})
df.head()


# All modules are showing the similar result when we compare with actual value.So here we can take any model. 

# # Classification Report

# In[108]:


# Logistic Regression

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predict1))


# In[55]:


# SVR

from sklearn.metrics import classification_report
print(classification_report(y_test,predict2))


# In[112]:


#GradiantBoosting 

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predict3))


# In[113]:


# RandomForest

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predict4))


# # Confusion Matrix

# In[114]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[115]:


cm=confusion_matrix(y_test,predict1)
cm


# In[116]:


### accuracy


print("Accuracy_Score of Logistic Regression",round(accuracy_score(y_test,predict1),2))

print("Accuracy_Score of SVR", round(accuracy_score(y_test,predict2),2))  

print("Accuracy_Score of XGBoost", round(accuracy_score(y_test,predict3),2))  

print("Accuracy_Score of RandomForest", round(accuracy_score(y_test,predict4),2))

 


# # Precision

# In[117]:


print("Precision score of Logistic Regression",  round(precision_score(y_test,predict1), 2))

print("Precision score of SVR",  round(precision_score(y_test,predict2), 2))

print("Precision score of XGboost",  round(precision_score(y_test,predict3), 2))

print("Precision score of randomForest",  round(precision_score(y_test,predict4), 2))


# # Recall

# In[120]:


print("recall_score Logistic Regression",  round(recall_score(y_test,predict1), 2))

print("recall_score SVR ",round(recall_score(y_test,predict2), 2))

print("recall_score XGboostClassifier ",round(recall_score(y_test,predict3), 2))

print("recall_score RandomForestClassifier ",round(recall_score(y_test,predict4), 2))




# # F1 Score

# In[121]:


from sklearn.metrics import f1_score
print("F1 score of logistic regression is ",round(f1_score(y_test,predict1), 2))

print("F1 score of SVR is ",round(f1_score(y_test,predict2), 2))

print("F1 score of XGBoost is ",round(f1_score(y_test,predict3), 2))

print("F1 score of randomForest is ",round(f1_score(y_test,predict4), 2))



# RandomForestClassifier and GradientboostingClassifier we can take on the basis of above accuracy score. 

# # Hypertunning By GridsearchCV

# In[122]:


from sklearn.model_selection import GridSearchCV
pgradintboost={"loss":["log_loss"],"max_depth":[3,4,5],"n_estimators":[100,200,300,400]}


# In[123]:


# gradientboosting
grid2=GridSearchCV(GradientBoostingClassifier(),param_grid=pgradintboost,verbose=3)
grid2.fit(X_train,y_train)


# In[126]:


# print best parameter after tuning
print(grid2.best_params_)


# In[129]:


gbr=GradientBoostingClassifier(n_estimators= 100,loss= 'log_loss',max_depth=3)
gbr.fit(X_train,y_train)
prdiction71=gbr.predict(X_test)


# In[131]:


print("Accuaracy score of Gridsearchcv of XGBoost is ",round(accuracy_score(y_test,prdiction71),2))

print("recall_score Gridsearchcv XGboost ",round(recall_score(y_test,prdiction71), 2))

print("Precion_Score of Gridsearchcv XGboost",round(precision_score(y_test,prdiction71), 2))

print("F1 score of Gridsearchcv XGBoost is ",round(f1_score(y_test,prdiction71), 2))


# # GridSearchCV for RandomForestClassifier
# 

# In[162]:


prandomforest={"criterion":["entropy"],"max_depth":[3,4,5],"n_estimators":[100,200,300]}


grid21=GridSearchCV(RandomForestClassifier(),param_grid=prandomforest,verbose=3)
grid21.fit(X_train,y_train)


# In[155]:


# print best parameter after tuning
print(grid21.best_params_)


# In[160]:


GSER=RandomForestClassifier(n_estimators= 100,criterion='entropy',max_depth=3)
GSER.fit(X_train,y_train)
prdiction71=GSER.predict(X_test)


# In[161]:


print("Accuaracy score of Gridsearchcv of random is ",round(accuracy_score(y_test,prdiction71),2))

print("recall_score  Gridsearchcv of random is ",round(recall_score(y_test,prdiction71), 2))

print("Precision_Score Gridsearchcv of random is",round(precision_score(y_test,prdiction71), 2))

print("F1 score of Gridsearchcv random is ",round(f1_score(y_test,prdiction71), 2))


# # Check the performance of model By log loss

# In[138]:


# RandomforestClassifier
from sklearn.metrics import log_loss
logloss = log_loss(y_test,rc.predict_proba(X_test)) 
logloss


# In[142]:


#GradientBoostingClassifier
from sklearn.metrics import log_loss
logloss = log_loss(y_test,xg.predict_proba(X_test)) 
logloss


# From the above observations we can take the GradientBoostingClassifier as machin learning model for the dataset to prdict the credit card.
# 

# In[152]:


xyz={"GENDER":1,"CHILDREN":0,"Type_Income":0,"Marital_status":1,"Family_Members":2}
Sample=pd.DataFrame(xyz,index=[0])
Sample


# In[153]:


new=xg.predict(Sample)
print(new)


# So our initial Hypothesis regarding the annual income,Property Ownership,Children,Gender,Employment history are wrong.
# Machine learning models are not taking into consideration in the approval or disapproval of credit card.
