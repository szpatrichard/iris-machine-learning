"""
Machine learning example with Iris flower dataset
"""

# Import modules
from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Load dataset
URL = "./dataset/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(URL, names=names)

# 2. Summarise dataset

## 2.1 Dimensions of dataset
print(dataset.shape) # (150 instances, 5 attributes)

## 2.2 Peek at the data
print(dataset.head(10)) # shows first 10 instances
print(dataset.tail(10)) # shows last 10 instances

## 2.3 Statistical summary
print(dataset.describe()) # describes dataset including with summary statistics

## 2.4 Class distribution
print(dataset.groupby("class").size()) # shows number of instances that belong to each class

# 3. Data visualisation

## 3.1 Univariate plots (for better understanding each attribute)

### 3.1.1 Box plot
dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.savefig("./figures/boxplot")

### 3.1.2. Histograms
dataset.hist()
pyplot.savefig("./figures/histogram")

## 3.2 Multivariate plots (for better understanding the relationships between attributes)

### 3.2.1 Scatter plots
scatter_matrix(dataset)
pyplot.savefig("./figures/scatter")

## 4. Evaluate algorithms

### 4.1 Create a validation set
array = dataset.values
X = array[:, 0:4] # select first 4 columns of every row
Y = array[:, 4] # select the last column of every row (class)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=1)

### 4.2 Build models
##### Spot Check Algorithms
models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))

##### Evaluate each model
results = []
alg_names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    alg_names.append(name)
    print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})")

## 5. Predictions

### 5.1 Make predictions
model = SVC(gamma="auto")
model.fit(X_train, Y_train)
predictions = model.predict((X_validation))

### 5.2 Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

## 6. Connect model input data with predictions
new_input = [[6.4, 3.6, 4.8, 1.4]]
new_output = model.predict(new_input)

print(new_input, new_output)
