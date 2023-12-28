from sklearn import tree
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(402)

# Set font for matplotlib
import matplotlib
font = {'weight' : 'normal', 'size' : 18}
matplotlib.rc('font', **font)

# Load the breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split into train and test 60/40
X_train = X[:len(X)//5*3]
y_train = y[:len(X)//5*3]
X_test = X[len(X)//5*3:]
y_test = y[len(X)//5*3:]

# Train a single decision tree on this dataset and store the accuracy
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
clf = clf.fit(X_train, y_train) 
single_acc = clf.score(X_test, y_test)

# Number of models in the ensemble
M = 15

# Initialize predictions array
preds = np.zeros((X_test.shape[0], M))

# Bagging: Train an ensemble with 15 models
for m in range(M):
    # Sample data with replacement for bagging
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bagged = X_train[idx]
    y_bagged = y_train[idx]

    # Fit the model and store its predictions
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=np.random.randint(10000))
    clf = clf.fit(X_bagged, y_bagged)
    preds[:, m] = clf.predict(X_test)

# Compute ensemble outputs and accuracy
ensemble_preds = np.mean(preds, axis=1) > 0.5
ensemble_acc = np.mean(ensemble_preds == y_test)

# Compute correlation between models
C = np.zeros((M, M))
for i in range(M):
    for j in range(i, M):
        C[i, j] = np.corrcoef(preds[:, i], preds[:, j])[0, 1]

# Make a pretty plot
C[C == 0] = np.nan
plt.figure(figsize=(12, 12))
plt.imshow(C, vmin=0, vmax=1)
plt.grid(True, alpha=0.15)
plt.text(M*0.05, M*0.92, "Single Model Acc. = {:2.3f}".format(single_acc), fontsize=24)
plt.text(M*0.05, M*0.78, "Avg Corr. = {:2.3f}".format(np.nanmean(C)), fontsize=24)
plt.text(M*0.05, M*0.85, "Ensemble Acc. = {:2.3f}".format(ensemble_acc), fontsize=24)
plt.colorbar()
plt.yticks(np.arange(0, M))
plt.xticks(np.arange(0, M))
plt.xlabel("Model ID")
plt.ylabel("Model ID")
plt.title("Prediction Correlation Between Models (Bagging) ")
plt.show()

# Number of models in the ensemble
M = 15

# Initialize predictions array
preds = np.zeros((X_test.shape[0], M))

# Train an ensemble with 15 models using max_features
for m in range(M):
    # Set max_features to a value less than the number of features
    # You can experiment with different values here
    max_features = 3

    # Fit the model and store its predictions
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_features=max_features, random_state=np.random.randint(10000))
    clf = clf.fit(X_train, y_train)
    preds[:, m] = clf.predict(X_test)

# Compute ensemble outputs and accuracy
ensemble_preds = np.mean(preds, axis=1) > 0.5
ensemble_acc = np.mean(ensemble_preds == y_test)

# Compute correlation between models
C = np.zeros((M, M))
for i in range(M):
    for j in range(i, M):
        C[i, j] = np.corrcoef(preds[:, i], preds[:, j])[0, 1]

# Make a pretty plot
C[C == 0] = np.nan
plt.figure(figsize=(12, 12))
plt.imshow(C, vmin=0, vmax=1)
plt.grid(True, alpha=0.15)
plt.text(M*0.05, M*0.92, "Single Model Acc. = {:2.3f}".format(single_acc), fontsize=24)
plt.text(M*0.05, M*0.78, "Avg Corr. = {:2.3f}".format(np.nanmean(C)), fontsize=24)
plt.text(M*0.05, M*0.85, "Ensemble Acc. = {:2.3f}".format(ensemble_acc), fontsize=24)
plt.colorbar()
plt.yticks(np.arange(0, M))
plt.xticks(np.arange(0, M))
plt.xlabel("Model ID")
plt.ylabel("Model ID")
plt.title(" Max Feature Selection - Prediction Correlation Between Models")
plt.show()
