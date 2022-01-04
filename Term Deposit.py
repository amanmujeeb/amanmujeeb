#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Libraries to split data, impute missing values 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Libtune to tune model, get different metric scores
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


# # Load data

# In[2]:


df=pd.read_csv("bank-full.csv")


# In[3]:


df.head()


# In[4]:


df.groupby("job").mean()["balance"]


# In[5]:


df.info()


# We See that there are no null values, there are 17 columns and 45211 observations

# # Checking unique values

# In[6]:


df.nunique()


# # Summarising the data

# In[7]:


df.describe().T


# Mean and median age is close, i.e. 41 and 39
# 
# Balance has outliers, need to explpore further
# 
# Duration has outliers, need to explore further
# 
# The campaign number of contacts has outliers as well, need to explore further
# 
# Pdays and previous has outliers as well

# # Unique categoriers in categoru columns

# In[8]:


#Making a list of all catrgorical variables 
cat_col=['job', 'marital','education', 'default', 'housing',
       'loan', 'contact', 'month',
        'poutcome', 'Target']

#Printing number of count of each unique value in each column
for column in cat_col:
    print(df[column].value_counts())
    print('-'*50)


# Most of the customers are married with no personal loan, but with housing loans

# # Converting to categories

# In[9]:


for column in cat_col:
    df[column]=df[column].astype('category')


# In[10]:


df.info()


# In[11]:



def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[12]:


histogram_boxplot(df, "age")


# Is almost normally distibuted, but the age does have outliers

# In[13]:


histogram_boxplot(df, "balance")


# Balance is normally distributed, but skewed with outliers

# In[14]:


histogram_boxplot(df, "pdays")


# In[15]:


histogram_boxplot(df, "previous")


# Both Pdays and previous are not normally distributed and are skewed

# # Dropping Outliers

# In[16]:


df[(df.age>80)]


# In[17]:


df[(df.balance>60000)]


# In[18]:


#Dropping observaions with age over 80, there are 99 observations
df.drop(index=df[df.age>80].index,inplace=True)

#Dropping observation with balance more than 60000
df.drop(index=df[df.balance>60000].index,inplace=True)


# # Function to perform EDA for categories

# In[19]:



def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  


# In[20]:


labeled_barplot(df, "job", perc=True)


# We see that the most number of customers belong to the blue collar category followed by management and technicians.

# In[21]:


labeled_barplot(df, "marital", perc=True)


# Most of the customers, around 60.2% are married

# In[22]:


labeled_barplot(df, "education", perc=True)


# 51.4% of the customers have completed secondary education.

# In[23]:


labeled_barplot(df, "default", perc=True)


# 98.2% of the customers are not defaulters for a loan

# In[24]:


labeled_barplot(df, "housing", perc=True)


# Around 55.7% of the customers take out housing loans

# In[25]:


labeled_barplot(df, "loan", perc=True)


# 83.9% of the customers do not have personal loans

# In[26]:


labeled_barplot(df, "contact", perc=True)


# Most of the customers are contacted via a cellurar channel

# In[27]:


labeled_barplot(df, "poutcome", perc=True)


# The outcome for most of the customers is unkown

# In[28]:


labeled_barplot(df, "Target", perc=True)


# 11.6% is the success rate of converion in order to get the client subscribed to a term deposit
# 
# The plot also shows that the distribution of both classes is imbalanced

# # Bivariative Analysis

# In[29]:


sns.pairplot(data=df,hue='Target')


# # Defining Functions

# In[30]:



def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# # Form of contact vs term deposit taken

# In[74]:


stacked_barplot(df, "contact", "Target" )


# The conversion rate of the people contatced via cellular support is the highest. 

# # Job vs Conversion

# In[75]:


stacked_barplot(df, "job", "Target" )


# We see that the conversion rate of studnets for the term loan is the highest out of all occupations followed by retired

# # Marriage Status vs conversion 

# In[76]:


stacked_barplot(df, "marital", "Target" )


# Our conversion rate for single people is the highest followed by divorced people

# 
# # Target Vs Balance Vs Job

# In[85]:


plt.figure(figsize=(15,5))
sns.boxplot(y='balance',x='job',hue='Target',data=df)
plt.show()


# # Replacing and converting yes/no to string

# In[31]:


df.default.replace(('yes', 'no'), (1, 0), inplace=True)
df.housing.replace(('yes', 'no'), (1, 0), inplace=True)
df.loan.replace(('yes', 'no'), (1, 0), inplace=True)
df.Target.replace(('yes', 'no'), (1, 0), inplace=True)


# # Correlation Heatmap

# In[32]:


sns.set(rc={'figure.figsize':(7,7)})
sns.heatmap(df.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="Spectral",
            fmt='0.2f')
plt.show()


# # Split the train test data

# In[33]:


X=df.drop(columns='Target')
Y=df['Target']


# ## We must drop month, day, pdays and previous 

# In[34]:


X.drop(columns=['month','day','duration','pdays','previous'],inplace=True)


# In[35]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=1,stratify=Y)


# In[36]:


for column in [ 'default','housing','loan']:
    X_train[column]=X_train[column].astype('float')
    X_test[column]=X_test[column].astype('float')


# In[37]:


col_dummy=['job', 'marital','education', 'contact', 
        'poutcome']


# In[38]:


X_train=pd.get_dummies(X_train, columns=col_dummy, drop_first=True)
X_test=pd.get_dummies(X_test, columns=col_dummy, drop_first=True)


# In[39]:


X_train.info()


# 
# # Build Model

# We need the metric precision to be the highest as the most of the conversion depends on how results of the previous campaigns

# In[40]:


def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

   
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  
    recall = recall_score(target, pred)  
    precision = precision_score(target, pred)
    f1 = f1_score(target, pred) 


    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf


# In[41]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# # Decision Tree

# In[42]:


d_tree = DecisionTreeClassifier(random_state=1)
d_tree.fit(X_train,y_train)
d_tree_model_train_perf=model_performance_classification_sklearn(d_tree, X_train,y_train)
print("Training performance:\n", d_tree_model_train_perf)
d_tree_model_test_perf=model_performance_classification_sklearn(d_tree, X_test,y_test)
print("Testing performance:\n", d_tree_model_test_perf)
confusion_matrix_sklearn(d_tree,X_test,y_test)


# Model is overfitting as training recall and precision are significantly higher

# # Cost Complexity Pruining

# In[43]:


path = d_tree.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[44]:


clfs_list = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs_list.append(clf)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs_list[-1].tree_.node_count, ccp_alphas[-1]))


# In[45]:


recall_train=[]
for clf in clfs_list:
    pred_train=clf.predict(X_train)
    values_train=metrics.recall_score(y_train,pred_train)
    recall_train.append(values_train)


# In[46]:


recall_test=[]
for clf in clfs_list:
    pred_test=clf.predict(X_test)
    values_test=metrics.recall_score(y_test,pred_test)
    recall_test.append(values_test)


# In[47]:


fig, ax = plt.subplots(figsize=(15,5))
ax.set_xlabel("alpha")
ax.set_ylabel("Recall")
ax.set_title("Recall vs alpha for training and testing sets")
ax.plot(ccp_alphas, recall_train, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, recall_test, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[48]:


index_best_pruned_model = np.argmax(recall_test)
pruned_dtree_model = clfs_list[index_best_pruned_model]
pruned_dtree_model_train_perf=model_performance_classification_sklearn(pruned_dtree_model, X_train,y_train)
print("Training performance:\n", pruned_dtree_model_train_perf)
pruned_dtree_model_test_perf=model_performance_classification_sklearn(pruned_dtree_model, X_test,y_test)
print("Testing performance:\n", pruned_dtree_model_test_perf)
confusion_matrix_sklearn(pruned_dtree_model,X_test,y_test)


# The model still hasnt improved as it is overfitting on the training set

# # Hyperparameter Tuning

# In[49]:


dtree_estimator = DecisionTreeClassifier(class_weight={0:0.12,1:0.88},random_state=1)
parameters = {'max_depth': np.arange(2,30), 
              'min_samples_leaf': [1, 2, 5, 7, 10],
              'max_leaf_nodes' : [2, 3, 5, 10,15],
              'min_impurity_decrease': [0.0001,0.001,0.01,0.1]
             }
scorer = metrics.make_scorer(metrics.recall_score)
grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)
dtree_estimator = grid_obj.best_estimator_
dtree_estimator.fit(X_train, y_train)


# In[50]:


dtree_estimator_model_train_perf=model_performance_classification_sklearn(dtree_estimator, X_train,y_train)
print("Training performance:\n", dtree_estimator_model_train_perf)
dtree_estimator_model_test_perf=model_performance_classification_sklearn(dtree_estimator, X_test,y_test)
print("Testing performance:\n", dtree_estimator_model_test_perf)
confusion_matrix_sklearn(dtree_estimator,X_test,y_test)


# The model is not overfitting in the data

# # Random Forest Classifier

# In[51]:


rf_estimator = RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train,y_train)
rf_estimator_model_train_perf=model_performance_classification_sklearn(rf_estimator, X_train,y_train)
print("Training performance:\n",rf_estimator_model_train_perf)
rf_estimator_model_test_perf=model_performance_classification_sklearn(rf_estimator, X_test,y_test)
print("Testing performance:\n",rf_estimator_model_test_perf)
confusion_matrix_sklearn(rf_estimator,X_test,y_test)


# This model performs better for precision and poor on recall.
# 
# We also see that the model is overfitting on training set

# # Hyperparameter Tuning

# In[52]:


rf_tuned = RandomForestClassifier(class_weight={0:0.12,1:0.88},random_state=1,oob_score=True,bootstrap=True)
parameters = {  
                'max_depth': list(np.arange(5,30,5)) + [None],
                'max_features': ['sqrt','log2',None],
                'min_samples_leaf': np.arange(1,15,5),
                'min_samples_split': np.arange(2, 20, 5),
                'n_estimators': np.arange(10,110,10)}
grid_obj = GridSearchCV(rf_tuned, parameters, scoring='recall',cv=5,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)
rf_tuned = grid_obj.best_estimator_ 
rf_tuned.fit(X_train, y_train)


# In[53]:


rf_tuned_model_train_perf=model_performance_classification_sklearn(rf_tuned, X_train,y_train)
print("Training performance:\n",rf_tuned_model_train_perf)
rf_tuned_model_test_perf=model_performance_classification_sklearn(rf_tuned, X_test,y_test)
print("Testing performance:\n",rf_tuned_model_test_perf)
confusion_matrix_sklearn(rf_tuned,X_test,y_test)


# The model is not overfitting as much and the metrics are better

# # Bagging Classifier

# In[54]:


bagging_classifier = BaggingClassifier(random_state=1)
bagging_classifier.fit(X_train,y_train)
bagging_classifier_model_train_perf=model_performance_classification_sklearn(bagging_classifier, X_train,y_train)
print("Training performance:\n",bagging_classifier_model_train_perf)
bagging_classifier_model_test_perf=model_performance_classification_sklearn(bagging_classifier, X_test,y_test)
print("Testing performance:\n",bagging_classifier_model_test_perf)
confusion_matrix_sklearn(bagging_classifier,X_test,y_test)


#  The model is still overfitting and the precision is good in testing

# # Hyperparameter Tuning

# In[55]:


bagging_estimator_tuned = BaggingClassifier(random_state=1)
parameters = {'max_samples': [0.7,0.8,0.9,1], 
              'max_features': [0.7,0.8,0.9,1],
              'n_estimators' : [10,20,30,40,50],
             }
acc_scorer = metrics.make_scorer(metrics.recall_score)
grid_obj = GridSearchCV(bagging_estimator_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)
bagging_estimator_tuned = grid_obj.best_estimator_
bagging_estimator_tuned.fit(X_train, y_train)


# In[56]:


bagging_estimator_tuned_model_train_perf=model_performance_classification_sklearn(bagging_estimator_tuned, X_train,y_train)
print("Training performance:\n",bagging_estimator_tuned_model_train_perf)
bagging_estimator_tuned_model_test_perf=model_performance_classification_sklearn(bagging_estimator_tuned, X_test,y_test)
print("Testing performance:\n",bagging_estimator_tuned_model_test_perf)
confusion_matrix_sklearn(bagging_estimator_tuned,X_test,y_test)


# The precision metric is performing better on the test set

# # AdaBoost Classifier

# In[57]:


ab_classifier = AdaBoostClassifier(random_state=1)
ab_classifier.fit(X_train,y_train)
ab_classifier_model_train_perf=model_performance_classification_sklearn(ab_classifier, X_train,y_train)
print("Training performance:\n",ab_classifier_model_train_perf)
ab_classifier_model_test_perf=model_performance_classification_sklearn(ab_classifier, X_test,y_test)
print("Testing performance:\n",ab_classifier_model_test_perf)
confusion_matrix_sklearn(ab_classifier,X_test,y_test)


# The model is not overfitting and is performing well in the precision metric on both sets

# # Hyperparameter Tuning

# In[58]:


abc_tuned = AdaBoostClassifier(random_state=1)
parameters = {
    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),
                      DecisionTreeClassifier(max_depth=3)],
    "n_estimators": np.arange(10,110,10),
    "learning_rate":np.arange(0.1,2,0.1)
}
acc_scorer = metrics.make_scorer(metrics.recall_score)
grid_obj = GridSearchCV(abc_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)
abc_tuned = grid_obj.best_estimator_
abc_tuned.fit(X_train, y_train)


# In[59]:


abc_tuned_model_train_perf=model_performance_classification_sklearn(abc_tuned, X_train,y_train)
print("Training performance:\n",abc_tuned_model_train_perf)
abc_tuned_model_test_perf=model_performance_classification_sklearn(abc_tuned, X_test,y_test)
print("Testing performance:\n",abc_tuned_model_test_perf)
confusion_matrix_sklearn(abc_tuned,X_test,y_test)


# The model is not overfitting but the precision metrics perform differently on train and test set

# # Gradient Boosting Classifier

# In[60]:


gb_classifier = GradientBoostingClassifier(random_state=1)
gb_classifier.fit(X_train,y_train)
gb_classifier_model_train_perf=model_performance_classification_sklearn(gb_classifier, X_train,y_train)
print("Training performance:\n",gb_classifier_model_train_perf)
gb_classifier_model_test_perf=model_performance_classification_sklearn(gb_classifier, X_test,y_test)
print("Testing performance:\n",gb_classifier_model_test_perf)
confusion_matrix_sklearn(gb_classifier,X_test,y_test)


# In[ ]:





# # Hyperparameter Tuning

# In[61]:


gbc_tuned = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),random_state=1)
parameters = {
    "n_estimators": [100,150,200,250],
    "subsample":[0.8,0.9,1],
    "max_features":[0.7,0.8,0.9,1]
}
acc_scorer = metrics.make_scorer(metrics.recall_score)
grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)
gbc_tuned = grid_obj.best_estimator_
gbc_tuned.fit(X_train, y_train)


# In[62]:


#Calculating different metrics
gbc_tuned_model_train_perf=model_performance_classification_sklearn(gbc_tuned, X_train,y_train)
print("Training performance:\n",gbc_tuned_model_train_perf)
gbc_tuned_model_test_perf=model_performance_classification_sklearn(gbc_tuned, X_test,y_test)
print("Testing performance:\n",gbc_tuned_model_test_perf)

#Creating confusion matrix
confusion_matrix_sklearn(gbc_tuned,X_test,y_test)


# The model is not overfitting, and the model has good precision

# # Comparing all models

# In[70]:


models_train_comp_df = pd.concat(
    [d_tree_model_train_perf.T,pruned_dtree_model_train_perf.T, dtree_estimator_model_train_perf.T, rf_estimator_model_train_perf.T,
    rf_tuned_model_train_perf.T,bagging_classifier_model_train_perf.T,bagging_estimator_tuned_model_train_perf.T,ab_classifier_model_train_perf.T,
     abc_tuned_model_train_perf.T,gb_classifier_model_train_perf.T,gbc_tuned_model_train_perf.T],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Prunned Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Bagging Classifier",
    "Bagging Estimator Tuned",
    "Adaboost Classifier",
    "Adaboost Classifier Tuned",
     "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned"]
models_train_comp_df


# In[66]:


# Testing performance comparison

models_test_comp_df = pd.concat(
    [d_tree_model_test_perf.T,pruned_dtree_model_test_perf.T, dtree_estimator_model_test_perf.T, rf_estimator_model_test_perf.T,
    rf_tuned_model_test_perf.T,bagging_classifier_model_test_perf.T,bagging_estimator_tuned_model_test_perf.T,ab_classifier_model_test_perf.T,
     abc_tuned_model_test_perf.T,gb_classifier_model_test_perf.T,gbc_tuned_model_test_perf.T],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree",
    "Prunned Decision Tree",
    "Decision Tree Estimator",
    "Random Forest Estimator",
    "Random Forest Tuned",
    "Bagging Classifier",
    "Bagging Estimator Tuned",
    "Adaboost Classifier",
    "Adaboost Classifier Tuned",
     "Gradient Boost Classifier",
    "Gradient Boost Classifier Tuned"]
print("Testing performance comparison:")
models_test_comp_df


# In[71]:


feature_names = X_train.columns
importances = gb_classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# Gradient Boost Classifier has the best precision. 
# 
# Previous outcome plays a big factor in success, followed by age and the housing loan

# # Reccomendations

# We see that a lot of the people who are inclined towards term deposits are students. We should choose to increase the enegement by running campaigns to increase the audience
# 
# We see that the age group of the customers and their success rate is corellated.
# 
# While conducting the campaign, the time and day of call and other external facrtors should be accounted as well
# 
# 
