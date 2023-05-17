# Diabetes Prediction

## Description
The Diabetes Prediction project aims to develop a machine learning model that can accurately predict the presence or absence of diabetes in individuals based on various clinical features. The project utilizes a dataset containing information such as glucose levels, blood pressure, body mass index (BMI), age, and other factors related to diabetes.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Cleaning](#data-cleaning)
- [Data Visualization](#data-visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run the code and reproduce the results, follow these steps:
1. Clone this repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage
To execute the code, open the project in your Python environment and run the main script or individual code sections.

## Data Preprocessing
The first step in the data analysis process is to clean and prepare the dataset for further analysis. The following code segment performs data cleaning tasks:

- df.head()
is used to display the top 5 rows of a DataFrame 

- df.columns 
learning about the columns

- df.info()
Print a concise summary of a DataFrame.

- df.describe()
helps us to understand how data has been spread across the table.

## Data Cleaning
Data cleaning is an essential step in preparing the dataset for analysis and modeling. The following data cleaning tasks are performed:


- df = df.drop_duplicates()
 Dropping Duplicates

- df.isnull().sum()
is used to count the number of missing values in each column of a DataFrame

- print(df[df['BloodPressure']==0].shape[0])
- print(df[df['Glucose']==0].shape[0])
- print(df[df['SkinThickness']==0].shape[0])
- print(df[df['Insulin']==0].shape[0])
- print(df[df['BMI']==0].shape[0])
checking for 0 values in 5 columns , 

-- Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace ,
#also no. of pregnancies as 0 is possible as observed in df.describe

Data cleaning is an essential step in preparing the dataset for analysis and modeling. The following data cleaning tasks are performed:

- Dropping Duplicates:
```python
df = df.drop_duplicates()
The code removes any duplicate rows from the dataset, ensuring that each record is unique and preventing redundant information from affecting the analysis.

Handling Missing Values:
python
Copy code
df.isnull().sum()
The code checks for missing values in the dataset and prints the sum of missing values for each column. This step helps identify columns with missing data, which can then be handled appropriately, such as through imputation or removal of missing values.
```
## Data Visualization
To gain a better understanding of the distribution of each feature in the dataset, histograms are generated. This provides insights into the data range and helps identify any potential outliers or patterns.
```python
#histogram for each  feature
df.hist(bins=10,figsize=(10,10))

plt.show()
plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x='Glucose',data=df)
plt.subplot(3,3,2)
sns.boxplot(x='BloodPressure',data=df)
plt.subplot(3,3,3)
sns.boxplot(x='Insulin',data=df)
plt.subplot(3,3,4)
sns.boxplot(x='BMI',data=df)
plt.subplot(3,3,5)
sns.boxplot(x='Age',data=df)
plt.subplot(3,3,6)
sns.boxplot(x='SkinThickness',data=df)
plt.subplot(3,3,7)
sns.boxplot(x='Pregnancies',data=df)
plt.subplot(3,3,8)
sns.boxplot(x='DiabetesPedigreeFunction',data=df)
```

### The code above calculates the correlation matrix using df.corr(), which computes the pairwise correlation between all columns in the dataset. The resulting correlation matrix, corrmat, represents the strength and direction of the linear relationship between the features.

```python
corrmat=df.corr()
sns.heatmap(corrmat, annot=True)
```
- The sns.heatmap() function then visualizes the correlation matrix as a heatmap. Each cell in the heatmap represents the correlation between two features, with color indicating the strength of the correlation. The annot=True parameter enables the display of correlation values inside each cell.

- This heatmap is useful for identifying features that are strongly correlated (either positively or negatively). High positive correlation suggests that the features move in the same direction, while high negative correlation indicates they move in opposite directions. It helps to identify potential multicollinearity issues and provides insights into the interdependencies between features.

- The generated correlation heatmap helps in feature selection, identifying redundant features, and understanding the relationships among variables in the dataset.

### Feature Transformation:
```python
from sklearn.preprocessing import QuantileTransformer

x = df_selected
quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new = pd.DataFrame(X)
df_new.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']
```

## Data Visualization
To visually explore the distribution and identify potential outliers in selected features, boxplots are generated using the `sns.boxplot()` function from the Seaborn library.

```python
plt.figure(figsize=(16, 12))
sns.set_style(style='whitegrid')
plt.subplot(3, 3, 1)
sns.boxplot(x=df_new['Glucose'], data=df_new)
plt.subplot(3, 3, 2)
sns.boxplot(x=df_new['BMI'], data=df_new)
plt.subplot(3, 3, 4)
sns.boxplot(x=df_new['Age'], data=df_new)
plt.subplot(3, 3, 5)
sns.boxplot(x=df_new['SkinThickness'], data=df_new)
```
The code above creates a figure with a size of 16x12 inches using plt.figure(figsize=(16, 12)). The sns.set_style(style='whitegrid') line sets the style of the Seaborn plots to have a white grid background.
## Data Preparation and Splitting
To prepare the data for training a machine learning model, the dataset is split into independent features (X) and the dependent feature (y). The data is then divided into training and testing sets using the `train_test_split()` function from the scikit-learn library.

```python
target_name = 'Outcome'
y = df_new[target_name]  # Dependent variable
X = df_new.drop(target_name, axis=1)  # Independent variables

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Logistic Regression
Logistic regression is used as a classification algorithm to predict the outcome of the target variable based on the independent features. The scikit-learn library provides the `LogisticRegression()` class for logistic regression modeling.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

reg = LogisticRegression()
reg.fit(X_train, y_train)

lr_pred = reg.predict(X_test)

print("Classification Report:\n", classification_report(y_test, lr_pred))
print("\nF1 Score:", f1_score(y_test, lr_pred))
print("Precision Score:", precision_score(y_test, lr_pred))
print("Recall Score:", recall_score(y_test, lr_pred))

print("\nConfusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test, lr_pred))
```
- redictions are made on the test set using reg.predict(X_test), and the classification report is printed using classification_report(y_test, lr_pred). Additional evaluation metrics such as F1 score, precision score, and recall score are calculated using the corresponding functions from the scikit-learn library.

- Finally, a confusion matrix is generated using confusion_matrix(y_test, lr_pred) and visualized as a heatmap using sns.heatmap().

- The classification report, evaluation metrics, and confusion matrix provide insights into the performance of the logistic regression model, including precision, recall, F1 score, and accuracy.

## Decision Tree Classifier
A decision tree classifier is used to predict the outcome of the target variable based on the independent features. The scikit-learn library provides the `DecisionTreeClassifier()` class for decision tree modeling.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(random_state=42)

# Create the parameter grid based on the results of random search
params = {
    'max_depth': [5, 10, 20, 25],
    'min_samples_leaf': [10, 20, 50, 100, 120],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=dt, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")
best_model = grid_search.fit(X_train, y_train)

dt_pred = best_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, dt_pred))
print("\nF1 Score:", f1_score(y_test, dt_pred))
print("Precision Score:", precision_score(y_test, dt_pred))
print("Recall Score:", recall_score(y_test, dt_pred))

print("\nConfusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test, dt_pred))
```
- n the code above, a decision tree classifier model is instantiated using DecisionTreeClassifier(). The model is then trained on the training data using best_model = grid_search.fit(X_train, y_train), which performs a grid search to find the best hyperparameters for the decision tree classifier.

- Predictions are made on the test set using best_model.predict(X_test), and the classification report is printed using classification_report(y_test, dt_pred). Additional evaluation metrics such as F1 score, precision score, and recall score are calculated using the corresponding functions from the scikit-learn library.

- Finally, a confusion matrix is generated using confusion_matrix(y_test, dt_pred) and visualized as a heatmap using sns.heatmap().

## Evaluation metric
After making predictions with the logistic regression and decision tree models, it's important to assess their performance using various evaluation metrics. The scikit-learn library provides functions to calculate metrics such as accuracy, precision, recall, and F1 score.
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate Logistic Regression model
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

# Evaluate Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

# Print evaluation metrics
print("Logistic Regression Performance:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1 Score:", lr_f1)

print("\nDecision Tree Performance:")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
```
- In the code above, various evaluation metrics are calculated for both the logistic regression and decision tree models. The metrics being calculated include accuracy, precision, recall, and F1 score. These metrics provide insights into the performance of the models in terms of their overall accuracy, ability to correctly identify positive cases, and ability to capture all relevant positive cases.

- Make sure to place this section under the Model Training and Evaluation subsection in the README file. Feel free to modify the code explanations and formatting according to your project's requirements.
