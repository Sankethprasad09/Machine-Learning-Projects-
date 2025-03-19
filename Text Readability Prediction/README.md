# 📘 Predicting Text Readability Using Machine Learning

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

```diff
+  Dataset source: https://www.kaggle.com/c/commonlitreadabilityprize/data 📊
```

## 📝 Problem Statement
With the increasing availability of books, articles, and digital content, categorizing texts based on readability levels has become essential. Automated text classification can assist libraries, educators, and readers in selecting appropriate reading materials without manually assessing difficulty levels.

This project applies **machine learning and deep learning** techniques to estimate the readability of texts based on linguistic complexity, word difficulty, and sentence structure.

![](https://github.com/suhasmaddali/Images/blob/main/patrick-tomasso-Oaqk7qqNh_c-unsplash.jpg)

## 🚀 Machine Learning Approach
### 📌 Natural Language Processing (NLP)
- **Text processing**: Tokenization, stopword removal, and lemmatization.
- **Feature extraction**: Converting words into numerical representations.
- **Machine learning regression models** to predict readability scores.

### 📊 Vectorization Methods
To convert text into a machine-readable format, we use:
- **[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)**
- **[TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)**
- **[Word2Vec (Glove Embeddings)](http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)**
- **[TF-IDF Word2Vec](https://datascience.stackexchange.com/questions/28598/word2vec-embeddings-with-tf-idf)**

### 🤖 Machine Learning Models
Since readability scores are continuous values, regression models are employed:
- **[Neural Networks](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)**
- **[Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**
- **[K-Nearest Neighbors (KNN) Regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)**
- **[Partial Least Squares (PLS) Regression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)**
- **[Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)**
- **[Gradient Boosting Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)**

## 📈 Exploratory Data Analysis (EDA)
- **Assessing dataset composition**: Identifying key features that affect text difficulty.
- **Handling missing values**: Removing features with minimal impact on readability classification.
- **Visualizing feature distributions**: Understanding correlations between text features and difficulty scores.

## 📊 Model Performance and Results
- **Deep Neural Networks (DNNs)** performed best in predicting readability scores.
- **Gradient Boosting Decision Trees (GBDT)** showed strong predictive capabilities.
- **TF-IDF Word2Vec embeddings** yielded the best encoding strategy for text representation.

## 🎯 Outcomes
✅ **Automated readability classification** to assist educators and librarians.
✅ **Enhanced text difficulty prediction using NLP and ML models.**
✅ **Potential real-world application** in digital content filtering and accessibility.

## 🔮 Future Scope
- **Integration into text editors (e.g., Microsoft Word)** to assist writers in adjusting readability.
- **Expansion of dataset sources** to refine model accuracy.

## 🛠 How to Run the Project
### Prerequisites
- **Git** (Download from [here](https://git-scm.com/downloads))
- **Python (>=3.7)**
- **Jupyter Notebook**
- **Dependencies**: Install using
  ```sh
  pip install -r requirements.txt
  ```

### Steps to Clone and Execute
1. **Clone the repository**:
   ```sh
   git clone <REPOSITORY_LINK>
   ```
2. **Navigate to the project folder**:
   ```sh
   cd text-readability-prediction
   ```
3. **Launch Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
4. **Run the notebook** to analyze readability scores.

---
This project provides an efficient way to predict readability scores, improving accessibility and content classification. 🚀

