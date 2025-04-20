# 🏡 House Price Prediction Model

![House Prediction](https://i.pinimg.com/736x/0e/c2/94/0ec294ac6fc6e583ad648f9aefd01666.jpg)

A simple yet effective machine learning model to predict house prices based on provided features. The project uses a dataset of 100 entries to train and evaluate a regression model with a **71% accuracy**. All training code is available in `train.py`, and the model can be reused or modified for further improvement.

---

## 📁 Project Structure


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
