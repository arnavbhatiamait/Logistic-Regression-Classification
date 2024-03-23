# %% [markdown]
# Logistic Regression

# %% [markdown]
# Logistic Regression Provides Us a Probablity of occurance of the event

# %% [markdown]
# Importing The Liberaries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# importing The Data Set

# %%

df=pd.read_csv("Social_Network_Ads.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y

# %% [markdown]
# Spliting Data set Into test and training sets

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# %%
x_train

# %%
x_test


# %%
y_train

# %%
y_test

# %% [markdown]
# Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# %%
x_test

# %%
x_train

# %% [markdown]
# Training The Logistic Regression model On Training Set

# %%
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# %% [markdown]
# Prediction of a New Result

# %%
print(classifier.predict(sc.transform([[30,87000]])))

# %% [markdown]
# Predicting The Test Set Results

# %%
y_pred=classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# %% [markdown]
# Making The Confusion Matrix 

# %%
from sklearn.metrics import confusion_matrix,accuracy_score
cnf=confusion_matrix(y_test,y_pred)
print(cnf)
accuracy_score(y_test,y_pred)

# %% [markdown]
# Heatmap

# %%
import seaborn as sns
className=[0,1]
fig,ax=plt.subplots()
tick_marks=np.arange(len(className))
plt.xticks(tick_marks,className)
plt.yticks(tick_marks,className)
# heatmap
sns.heatmap(pd.DataFrame(cnf),annot=True,cmap="YlGnBu",fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion Matrix",y=1.1)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")


# %% [markdown]
# 

# %% [markdown]
# Classification Report

# %%
from sklearn.metrics import classification_report
target_names = ['Buy' , 'Dont Buy']
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown]
# Visualizing The Training Set Results

# %% [markdown]
# 

# %%
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# %% [markdown]
# Visulizing The Test Set Results

# %%
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


