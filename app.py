import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#Loading the model
model=tf.keras.models.load_model('regressor.h5')

#loading the encoders and scalars
with open('lbe.pkl','rb') as file:
    lbe=pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe=pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)

# streamlit

st.title("Estimating the salary of a customer based on his bank details")

# User input
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', lbe.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
#estimated_salary = st.number_input('Estimated Salary')
exited=st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Example input data
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'Exited': exited
}
data=pd.DataFrame([input_data])

#Label encodin gender
data['Gender']=lbe.transform(data['Gender'])
# One-hot encode 'Geography'
geo_encoded = ohe.transform([data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

input_data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)

input_data_scaled=scalar.transform(input_data)

# Predict churn
prediction=model.predict(input_data_scaled)

predict_proba=prediction[0][0]

st.write(f"The salary of the customer based on their details will be {predict_proba:.2f}")


