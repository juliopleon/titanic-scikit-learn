# Imports
# Data
import pandas as pd
import numpy as np
# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("titanic.csv")
data.info()
print(data.isnull().sum())

# Data Cleaning and Future Engineering
def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)

    # Convert gender
    df["Sex"] = df["Sex"].map({'male':1,'female':0})

    # Feature Engineering!
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FairBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeRange"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf],labels=False)
