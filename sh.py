import numpy as np
import pandas as pd  #Use for data analysis
import matplotlib.pyplot as plt #for graphs
import seaborn as sns #Statistical library for analysis 
import statsmodels.api as sm  #for test in python

Shark_Data = pd.read_csv("Shark.csv",encoding='latin-1')
print(Shark_Data)

# sns_plot2=sns.pairplot(Shark_Data) 
# image=plt.show() 
# sns_plot2.savefig('shark.png')

from sklearn.model_selection import train_test_split

np.random.seed(0) #assign value of zero to random variable
df_train, df_test = train_test_split(Shark_Data, train_size = 0.7 , test_size = 0.3, random_state = 0)
#print(df_train.shape)
#print(df_test.shape)
##print(Table)
#Table.to_excel('Desktop:\Python\Table.xlsx') 

# plt.figure(figsize = (16,10))
# fig1=sns.heatmap(df_train.corr(), annot = True , cmap="GnBu") 
# plt.show()
# fig2=fig1.get_figure().savefig('Shark_Correlation.png')

y_train = df_train.pop('Total Deal Amount')
print(y_train)
X_train = df_train
print(X_train)
#print(X_train)
#print(y_train)

#Train Model 1
X_train_1 = X_train[['Valuation Offered','Accepted Offer','Number of sharks in deal']]
X_train_1 = sm.add_constant(X_train_1)
model_1 = sm.OLS(y_train, X_train_1).fit() 
print(model_1.params)
print(model_1.summary())

print("Model2")
X_train_2 = X_train[['Accepted Offer']]
X_train_2 = sm.add_constant(X_train_2)
model_2 = sm.OLS(y_train, X_train_2).fit() 
print(model_2.params)
print(model_2.summary())

print("Model3")
X_train_3 = X_train[['Valuation Offered','Accepted Offer']]
X_train_3 = sm.add_constant(X_train_3)
model_3 = sm.OLS(y_train, X_train_3).fit() 
print(model_3.params)
print(model_3.summary())

print("Model4")
X_train_4 = X_train[['Valuation Offered','Number of sharks in deal']]
X_train_4 = sm.add_constant(X_train_4)
model_4 = sm.OLS(y_train, X_train_4).fit() 
print(model_4.params)
print(model_4.summary())

# #Train Model 2
# X_train_2 = X_train[['Number of sharks in deal','Received Offer']]
# print(X_train_2)
# X_train_2 = sm.add_constant(X_train_2)
# model_2 = sm.OLS(y_train, X_train_2).fit() 
# print(model_2.params)
# print(model_2.summary())

# #Train Model 3 
# X_train_3 = X_train[['Number of sharks in deal','Received Offer','Total Deal Equity']]
# X_train_3 = sm.add_constant(X_train_3)
# model_3 = sm.OLS(y_train, X_train_3).fit() 
# print(model_3.params)
# print(model_3.summary())


# #Train Model 4
# X_train_4 = X_train[['Number of sharks in deal','Received Offer','Total Deal Equity','Total Deal Amount']]
# X_train_4 = sm.add_constant(X_train_4)
# model_4 = sm.OLS(y_train, X_train_4).fit() 
# print(model_4.params)
# print(model_4.summary())

# # #Train Model 5 
# # X_train_5 = X_train[['Number of sharks in deal','Received Offer','Total Deal Amount']]
# # X_train_5 = sm.add_constant(X_train_5)
# # model_5 = sm.OLS(y_train, X_train_5).fit() 
# # print(model_5.params)
# # print(model_5.summary())

# #Importer
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_3.columns
vif['VIF'] = [variance_inflation_factor(X_train_3.values, i) for i in range(X_train_3.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False ) 
vif.reset_index(drop=True, inplace=True)
print(vif)

# VIF for Model 1
vif = pd.DataFrame()
vif['Features'] = X_train_1.columns
vif['VIF'] = [variance_inflation_factor(X_train_1.values, i) for i in range(X_train_1.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False ) 
vif.reset_index(drop=True, inplace=True)
print(vif)

# VIF for Model 2
vif = pd.DataFrame()
vif['Features'] = X_train_2.columns
vif['VIF'] = [variance_inflation_factor(X_train_2.values, i) for i in range(X_train_2.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False ) 
vif.reset_index(drop=True, inplace=True)
print(vif)

# VIF for Model 4
vif = pd.DataFrame()
vif['Features'] = X_train_4.columns
vif['VIF'] = [variance_inflation_factor(X_train_4.values, i) for i in range(X_train_4.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False ) 
vif.reset_index(drop=True, inplace=True)
print(vif)


print("Model4")
X_train_4 = X_train[['Valuation Offered','Number of sharks in deal']]
X_train_4 = sm.add_constant(X_train_4)
model_4 = sm.OLS(y_train, X_train_4).fit() 
print(model_4.params)
print(model_4.summary())