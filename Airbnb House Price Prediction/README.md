# 🏡 Airbnb House Price Prediction

## 📌 Introduction
Airbnb home price prediction is a crucial task in real estate and rental markets. Various factors such as **location, demand, property type, and amenities** influence the rental prices. By leveraging **machine learning**, we aim to create a predictive model to estimate home prices based on key features.

## 📏 Performance Metrics
Since this is a **regression problem**, we evaluate models using:
- **[Mean Squared Error (MSE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)**
- **[Mean Absolute Error (MAE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)**

## 🔍 Exploratory Data Analysis (EDA)
- **Outlier detection**: Identified and removed extreme price values.
- **Room type analysis**: Majority of users prefer entire homes over private/shared rooms.
- **Geographical distribution**: Manhattan has the highest number of Airbnb listings.

## 📊 Data Visualization Insights
- **Feature distributions and correlations** help in understanding market trends.
- **Neighborhood analysis** reveals pricing differences across different boroughs.
- **Room type preferences** affect pricing significantly.

## 🔢 Machine Learning Models
Several regression models were explored:
- **[Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)**
- **[Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)**
- **[Neural Networks](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)**

## 🏆 Key Outcomes
✅ Helps Airbnb hosts **optimize pricing** for better profitability.
✅ Assists users in **finding cost-effective stays** based on location and amenities.
✅ Potential **integration into Airbnb's pricing recommendation system**.

## 🔮 Future Enhancements
- **Incorporate additional features** such as **weather conditions** and **crime rate**.
- **Deploy model on cloud services** like **AWS** for real-time predictions.
- **Enhance accuracy** with deep learning-based price estimation models.

## 📥 How to Download and Run the Repository
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
   cd airbnb-home-price-prediction
   ```
3. **Launch Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
4. **Run the notebook** to train and test the model.

---
This project provides an insightful approach to **predicting Airbnb prices**, assisting both **hosts and users** in optimizing their rental experience. 🚀

