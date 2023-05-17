# Diabetes Prediction

## Description
The Diabetes Prediction project aims to develop a machine learning model that can accurately predict the presence or absence of diabetes in individuals based on various clinical features. The project utilizes a dataset containing information such as glucose levels, blood pressure, body mass index (BMI), age, and other factors related to diabetes.

| Function              | Description                                          |
|-----------------------|------------------------------------------------------|
| df.corr()             | Computes the correlation matrix of a DataFrame `df`. |
| QuantileTransformer() |  function creates an instance of the QuantileTransformer class, which is a data transformation technique for mapping the features of a dataset to a specified distribution. |


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

Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace ,
#also no. of pregnancies as 0 is possible as observed in df.describe

## Data Visualization
To gain a better understanding of the distribution of each feature in the dataset, histograms are generated. This provides insights into the data range and helps identify any potential outliers or patterns.

```python
df.hist(bins=10,figsize=(10,10))
plt.show()
plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
```
- hist(): This method is called on the DataFrame to generate histograms.
- bins=10: Specifies the number of bins or intervals to divide the data range into. In this case, each feature's histogram will have 10 bins.
- figsize=(10,10): Sets the size of the figure or plot in inches. In this case, the width and height of the figure are both set to 10 inches.
After generating the histograms, the code plt.show() is used to display the plot.

Additionally, the code plt.figure(figsize=(16,12)) creates a new figure with a larger size before generating the histograms. This line sets the size of the subsequent plot to 16 inches in width and 12 inches in height.

By using these code snippets, you can visualize the distribution and range of values for each feature in the DataFrame using histograms. It helps in understanding the data distribution and identifying any potential outliers or patterns.

## Correlation Matrix

```python
corrmat=df.corr()
sns.heatmap(corrmat, annot=True)
```
- The sns.heatmap() function then visualizes the correlation matrix as a heatmap. Each cell in the heatmap represents the correlation between two features, with color indicating the strength of the correlation. The annot=True parameter enables the display of correlation values inside each cell.

- This heatmap is useful for identifying features that are strongly correlated (either positively or negatively). High positive correlation suggests that the features move in the same direction, while high negative correlation indicates they move in opposite directions. It helps to identify potential multicollinearity issues and provides insights into the interdependencies between features.

- The generated correlation heatmap helps in feature selection, identifying redundant features, and understanding the relationships among variables in the dataset.

### Feature Transformation:
```python
quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']
```
- quantile = QuantileTransformer(): This line creates an instance of the QuantileTransformer class. The QuantileTransformer is a data transformation technique that maps the data to a uniform or a normal distribution.
- X = quantile.fit_transform(x): The fit_transform() method is used to fit the transformer to the data (x) and transform it. The fit_transform() method performs two steps: it first fits the transformer to the data to learn the parameters, and then transforms the data using those learned parameters. The transformed data is stored in the variable X.
- df_new.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']: This line assigns new column names to the DataFrame df_new. It sets the column names to 'Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', and 'Outcome' respectively.
Overall, the code uses the QuantileTransformer to transform the data in x, which is then stored in the variable X. It then assigns new column names to the DataFrame df_new. This transformation is often used to normalize the data and make it more suitable for certain machine learning algorithms that assume a specific data distribution.

## Data Preparation and Splitting
To prepare the data for training a machine learning model, the dataset is split into independent features (X) and the dependent feature (y). The data is then divided into training and testing sets using the `train_test_split()` function from the scikit-learn library.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
- X: This variable represents the input features or independent variables.
- y: This variable represents the target variable or dependent variable.
- test_size=0.2: This parameter specifies the proportion of the data that should be allocated to the testing set. In this case, 20% of the data will be used for testing, while the remaining 80% will be used for training the model.
random_state=0: This parameter sets the random seed for reproducibility. By setting it to a specific value (in this case, 0), the same train-test split will be obtained each time the code is executed.
- The train_test_split() function from scikit-learn is used to split the data into training and testing sets based on the provided parameters. It returns four sets of data:

- X_train: This variable contains the training data for the independent features.
- X_test: This variable contains the testing data for the independent features.
- y_train: This variable contains the training data for the target variable.
- y_test: This variable contains the testing data for the target variable.
The purpose of splitting the data into training and testing sets is to assess the performance of the machine learning model on unseen data. The model is trained on the training set (X_train and y_train), and then evaluated on the testing set (X_test and y_test) to measure its generalization ability.

## Logistic Regression
Logistic regression is used as a classification algorithm to predict the outcome of the target variable based on the independent features. The scikit-learn library provides the `LogisticRegression()` class for logistic regression modeling.

```python
reg = LogisticRegression()
reg.fit(X_train, y_train)
lr_pred = reg.predict(X_test)
```
- reg = LogisticRegression(): This line creates an instance of the LogisticRegression class. Logistic regression is a popular classification algorithm used to predict binary outcomes based on input features.
- reg.fit(X_train, y_train): The fit() method is called on the LogisticRegression object to train the model. It takes the -training data X_train (independent features) and y_train (target variable) as input. During this step, the logistic regression model learns the coefficients for the features that best fit the training data.
- lr_pred = reg.predict(X_test): After the model has been trained, the predict() method is used to make predictions on the testing data X_test. The logistic regression model assigns a class label (0 or 1) to each sample in the testing data based on its learned parameters and the input features. The predicted class labels are stored in the variable lr_pred.

By executing this code, you train a logistic regression model using the training data and make predictions on the testing data. The predictions are stored in lr_pred and can be used for further evaluation, such as calculating performance metrics or analyzing the accuracy of the model.

## Decision Tree Classifier
A decision tree classifier is used to predict the outcome of the target variable based on the independent features. The scikit-learn library provides the `DecisionTreeClassifier()` class for decision tree modeling.

```python
dt = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [5, 10, 20, 25],
    'min_samples_leaf': [10, 20, 50, 100, 120],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=dt, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")
best_model = grid_search.fit(X_train, y_train)

dt_pred = best_model.predict(X_test)
```
- dt = DecisionTreeClassifier(random_state=42): This line creates an instance of the DecisionTreeClassifier class, which is a classification algorithm based on decision trees. The random_state parameter is set to 42 to ensure reproducibility of results.
- grid_search = GridSearchCV(estimator=dt, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy"): This line creates an instance of the GridSearchCV class. It takes the DecisionTreeClassifier object dt, the parameter grid params, the number of cross-validation folds (cv=4), the number of parallel jobs to run (n_jobs=-1 means using all available cores), the verbosity level (verbose=1 for detailed output), and the scoring metric (scoring="accuracy" to evaluate models based on accuracy).

- best_model = grid_search.fit(X_train, y_train): The fit() method is called on the GridSearchCV object to perform the grid search with cross-validation. It trains and evaluates the decision tree models with different combinations of hyperparameters on the training data. The best model with the highest accuracy is selected based on cross-validation performance and stored in the best_model variable.
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
```
- In the code above, various evaluation metrics are calculated for both the logistic regression and decision tree models. The metrics being calculated include accuracy, precision, recall, and F1 score. These metrics provide insights into the performance of the models in terms of their overall accuracy, ability to correctly identify positive cases, and ability to capture all relevant positive cases.

- Make sure to place this section under the Model Training and Evaluation subsection in the README file. Feel free to modify the code explanations and formatting according to your project's requirements.
