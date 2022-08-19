#%% importing libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import plot_roc_curve, confusion_matrix, roc_auc_score, f1_score

import pickle

import warnings
warnings.filterwarnings("ignore")

# %%Reading in and inspecting data
df = pd.read_csv("../data/Tabla_01_English_Unique_postEDA.csv", index_col=0, parse_dates=True)
df.info(), df.shape

#%% Define X, y and split data (using complete dataset, i.e. years 2018 - may 2021 for fitting and evaluating models).
X = df[["LO_Active_Employee_Post3Months", "LO_Active_Employee_Prior6Months","LO_Active_Employee_Post6Months"]] # features selected based on calculation of feature importance in NB "all features".
y = df["Client_Status_Post3Months"]  
X.shape, y.shape

#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # no need to stratify (default includes it)

X_train.shape, y_train.shape, X_val.shape, y_val.shape


#%% Feature Engineering: creating time-related variables (training data)
X_train["Year"] = X_train.index.year
X_train["Month"] = X_train.index.month
X_train.columns

#%% Feature Engineering: creating scaling variables (training data)


scaler = MinMaxScaler()




Xtrain_scaled = scaler.fit_transform(X_train)


# In[13]:


X_train.shape


# ### Feature Engineering: validation data

# In[14]:


X_val["Year"] = X_val.index.year


# In[15]:


X_val["Month"] = X_val.index.month


# In[16]:


X_val.columns


# In[17]:


Xval_scaled = scaler.transform(X_val)


# In[18]:


X_val.shape


# ## Fit and evaluate models 

# ### Fit a Logistic Regression model

# In[19]:


# Define weights for cost-sensitive learning.
weights = {0:0.41, 1:0.59}     # rationale: taking inverse distribution of labels (see EDA on distribution of minority/majority groups).


# In[20]:


model_LR = LogisticRegression(class_weight = weights, random_state=42) 
model_LR.fit(X_train, y_train)


# ### Make predictions.

# In[21]:


ypred_LR = model_LR.predict(X_val)


# In[22]:


probs_LR = model_LR.predict_proba(X_val)
probs_LR


# ### Evaluate LogReg: ROC curve 

# In[23]:


model_LR_disp = plot_roc_curve(model_LR, X_val, y_val)
plt.show()


# ### Evaluate LogReg: confusion matrix and AUC

# In[24]:


print(f"Confusion Matrix: \n{confusion_matrix(y_val, ypred_LR)}")
print(f"Area Under Curve: {roc_auc_score(y_val, ypred_LR).round(2)}")
print(f"f1 - score: {f1_score(y_val, ypred_LR).round(2)}")


# Interpretation:
# - in contrast to the model LogReg model trained on all features, training the same one based on the most important features leads to clearly better results.
# - re Confusion Matrix: out of 529 predictions regarding the positive class, 86% were predicted correctly. Out of 332 predictions regarding the negative class, 9% were predicted incorrectly (false negatives).
# - f1 - score:  85%
# - re AUC: the likelihood that a randomly selected customer from the minority group is scored higher than the a randomly selected customer from the majority group is 88% (vs. model trained on all features: 50%).

# In[25]:


auc_LR = roc_auc_score(y_val, ypred_LR)
f1_LR = f1_score(y_val, ypred_LR)


# In[26]:


train_accuracy = model_LR.score(X_train, y_train)
train_accuracy


# In[27]:


val_accuracy = model_LR.score(X_val, y_val)
val_accuracy


# => while performance remains the same, with this dataset the model overfits less!

# ### Fit a Random Forest model

# In[28]:


model_RF = RandomForestClassifier(max_depth=10, min_samples_split=3, n_estimators=63, random_state=42) # params from manual tuning


# In[29]:


X_train.copy()
y_train.copy()


# In[30]:


model_RF.fit(X_train, y_train)


# ### Make predictions.

# In[31]:


ypred_RF = model_RF.predict(X_val)


# In[32]:


probs_RF = model_RF.predict_proba(X_val) 
probs_RF


# ### Evaluate Random Forest: ROC curve 

# In[33]:


ax = plt.gca()

model_RF_disp = plot_roc_curve(model_RF, X_val, y_val, ax=ax, alpha=0.8)
model_LR_disp.plot(ax=ax, alpha=0.8)

plt.show() 


# ### Evaluate Random Forest: confusion matrix and AUC

# In[34]:


print(f"Confusion Matrix: \n{confusion_matrix(y_val, ypred_RF)}")
print(f"Area Under Curve: {roc_auc_score(y_val, ypred_RF).round(2)}")
print(f"f1 - score: {f1_score(y_val, ypred_RF).round(2)}")


# In[35]:


auc_RF = roc_auc_score(y_val, ypred_RF)
f1_RF = f1_score(y_val, ypred_RF)


# Interpretation:
# - re Confusion Matrix: out of 529 predictions regarding the positive class, 100% were predicted correctly. Model seems to be overfitting!
# - Out of 332 predictions regarding the negative class, 12% were predicted incorrectly (false negatives).
# - f1 - score:  94% (vs. initially 93%)
# - re AUC: the likelihood that a randomly selected customer from the minority group is scored higher than the a randomly selected customer from the majority group is 94% - as in first version.

# In[36]:


train_accuracy = model_RF.score(X_train, y_train)
train_accuracy


# In[37]:


val_accuracy = model_RF.score(X_val, y_val)
val_accuracy


# => reduction in overfitting!

# ### Fit a Decision Tree model

# In[38]:


model_DT = DecisionTreeClassifier(max_depth=6, random_state=42) # param from manual tuning


# In[39]:


X_train.copy()
y_train.copy()


# In[40]:


model_DT.fit(X_train, y_train)


# ### Make predictions.

# In[41]:


ypred_DT = model_DT.predict(X_val)


# In[42]:


probs_DT = model_DT.predict_proba(X_val)
probs_DT


# ### Evaluate Decision Tree: ROC curve 

# In[43]:


ax = plt.gca()

model_DT_disp = plot_roc_curve(model_DT, X_val, y_val, ax=ax, alpha=0.8)
model_RF_disp.plot(ax=ax, alpha=0.8)
model_LR_disp.plot(ax=ax, alpha=0.8)

plt.show()


# ### Evaluate Decision Tree: confusion matrix and AUC

# In[44]:


print(f"Confusion Matrix: \n{confusion_matrix(y_val, ypred_DT)}")
print(f"Area Under Curve: {roc_auc_score(y_val, ypred_DT).round(2)}")
print(f"f1 - score: {f1_score(y_val, ypred_DT).round(2)}")


# In[45]:


auc_DT = roc_auc_score(y_val, ypred_DT)
f1_DT = f1_score(y_val, ypred_DT)


# Interpretation:
# - re Confusion Matrix: out of 332 predictions regarding the positive class, 90% were predicted correctly. As such, DT shows best precision of all models. Out of 529 predictions regarding the negative class, 6% were predicted incorrectly (false negatives). 
# - f1 - score = 89.95% 
# - re AUC: the likelihood that a randomly selected customer from the minority group is scored higher than the a randomly selected customer from the majority group is 92% in the case of a Decision Tree classifier (vs. 94% of a Random Forest classifier).

# In[46]:


train_accuracy = model_DT.score(X_train, y_train)
train_accuracy


# In[47]:


val_accuracy = model_DT.score(X_val, y_val)
val_accuracy


# => reduction in overfitting!

# ### Summary model evaluation: AUC

# In[48]:


auc = [["LogReg", auc_LR], ["Tree",  auc_DT], ["RF",  auc_RF]]
auc  = pd.DataFrame(auc , columns = ["Model", "auc"])
auc.sort_values(by=["auc"], inplace=True, ascending = False)
auc.set_index(["Model"])


# ### Summary model evaluation: f1 - score

# In[49]:


f1 = [["LogReg", f1_LR], ["Tree",  f1_DT], ["RF",  f1_RF]]
f1  = pd.DataFrame(f1 , columns = ["Model", "f1"])
f1.sort_values(by=["f1"], inplace=True, ascending = False)
f1.set_index(["Model"])


# #### Interpretation
# * Random Forest Model shows best performance (based on AUC and f1-score).
# * Both RF as well as Tree perform better than LogReg.
# * All models tend to overfit less than with the complete dataset.
# * LogReg suffers performance, in comparison with model trained on all features.

# ## Saving the Random Forest model.

# In[50]:


with open("../artefacts/churn-model.bin", "wb") as f_out:
    pickle.dump(model_RF, f_out) 

