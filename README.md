# 🏡 House Price Prediction Model

![House Prediction](https://i.pinimg.com/736x/0e/c2/94/0ec294ac6fc6e583ad648f9aefd01666.jpg)

This was a basic project to help me understand how machine learning models are trained. I created a simple house price prediction model using a dataset with 100 entries. The dataset included features like area (in square feet), number of bedrooms, and location. The target variable was the price (in lakhs). I found a basic dataset online and manually copied 100 entries into my working file.

The main goal of this project was to learn the steps of preprocessing and training a regression model.


---

## 🧠 Model Overview

- The model is trained using scikit-learn.
- All model definitions and training routines are included in `train.py`.
- Achieves **71% accuracy** on test data.
- Suitable as a lightweight regression baseline.

---

## 📊 Dataset

The dataset (`data.csv`) consists of 100 house entries with various features such as:

- Number of rooms  
- Area in square feet  
- Location score  
- Year built  
- ...and more  

Ensure that the dataset is in the same directory as `train.py` or modify the path accordingly.

---

## 📦 Installation

Install the required dependencies using the following command:

```bash
pip install pandas numpy joblib scikit-learn matplotlib
