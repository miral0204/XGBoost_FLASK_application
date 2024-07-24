from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/')
def index():
    # Load and process the dataset
    data = pd.read_csv('heart.csv')

    # Handle missing values
    data.isnull().sum()

    # Encode categorical variables
    lb = LabelEncoder()
    data.Sex = lb.fit_transform(data['Sex'])
    data.rename({'ChestPainType': 'cpt'}, axis=1, inplace=True)
    data.cpt.unique()
    data.loc[data['cpt'] == 'ASY', 'cpt'] = 3
    data.loc[data['cpt'] == 'NAP', 'cpt'] = 2
    data.loc[data['cpt'] == 'ATA', 'cpt'] = 1
    data.loc[data['cpt'] == 'TA', 'cpt'] = 0
    data.RestingECG.unique()
    data.loc[data['RestingECG'] == 'Normal', 'RestingECG'] = 2
    data.loc[data['RestingECG'] == 'LVH', 'RestingECG'] = 1
    data.loc[data['RestingECG'] == 'ST', 'RestingECG'] = 0
    data.ExerciseAngina = lb.fit_transform(data['ExerciseAngina'])
    data.ST_Slope.value_counts()
    data.loc[data['ST_Slope'] == 'Flat', 'ST_Slope'] = 2
    data.loc[data['ST_Slope'] == 'Up', 'ST_Slope'] = 1
    data.loc[data['ST_Slope'] == 'Down', 'ST_Slope'] = 0

    # Plot histograms
    plt.figure(figsize=(20, 25), facecolor='white')
    plotnumber = 1
    for column in data.columns:
        if plotnumber <= 9:
            ax = plt.subplot(3, 3, plotnumber)
            sns.histplot(x=data[column])
            plt.xlabel(column, fontsize=20)
        plotnumber += 1
    plt.tight_layout()
    plt.savefig('static/histogram.png')
    plt.close()

    # Plot histograms with Heart Disease hue
    plt.figure(figsize=(20, 25), facecolor='white')
    plotnumber = 1
    for column in data.columns:
        if plotnumber <= 9:
            ax = plt.subplot(3, 3, plotnumber)
            sns.histplot(x=data[column], hue=data.HeartDisease)
            plt.xlabel(column, fontsize=20)
            plt.ylabel('HeartDisease', fontsize=20)
        plotnumber += 1
    plt.tight_layout()
    plt.savefig('static/histogram_hue.png')
    plt.close()

    # Plot correlation heatmap
    plt.figure(figsize=(30, 30))
    sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":25})
    plt.xticks(fontsize=25, rotation=45)
    plt.yticks(fontsize=25, rotation=45)
    plt.tight_layout()
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

    # Creating X and y
    X = data.drop('HeartDisease', axis=1)
    y = data.HeartDisease

    # Creating training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    X_train.ST_Slope = X_train.ST_Slope.astype('int64')
    X_train.RestingECG = X_train.RestingECG.astype('int64')
    X_train.cpt = X_train.cpt.astype('int64')
    X_test.ST_Slope = X_test.ST_Slope.astype('int64')
    X_test.RestingECG = X_test.RestingECG.astype('int64')
    X_test.cpt = X_test.cpt.astype('int64')

    # Importing the model library
    gbm = GradientBoostingClassifier()  # Object creation
    gbm.fit(X_train, y_train)  # Fitting the data
    y_gbm = gbm.predict(X_test)  # Predicting the test data
    y_pre_train = gbm.predict(X_train)

    # Get classification reports
    train_report = classification_report(y_train, y_pre_train, output_dict=True)
    test_report = classification_report(y_test, y_gbm, output_dict=True)
    
    # Convert reports to HTML format
    train_report_html = pd.DataFrame(train_report).to_html()
    test_report_html = pd.DataFrame(test_report).to_html()

    return render_template('index.html', train_report=train_report_html, test_report=test_report_html)

if __name__ == '__main__':
    app.run(debug=True)
