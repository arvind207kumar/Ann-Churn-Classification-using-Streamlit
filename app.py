import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler , LabelEncoder
import pickle 


#Load the trained Model
model =tf.keras.models.load_model('model.h5')



### load the encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
 
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)    
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file) 


## Streamlit app
st.title('Customer Churn Prediction')



## user input 
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
Estimated_Salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_product = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_number = st.selectbox('Is Active Member',[0,1])


### Prepare the input data
input_data = pd.DataFrame({
'CreditScore': [credit_score],	
'Geography': [geography],
'Gender': [label_encoder_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_product],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_number],
'EstimatedSalary': [Estimated_Salary]	    
       
}
    
    
)



geo_encoder = onehot_encoder_geo.transform(input_data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),columns =onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df =pd.concat([input_data.drop('Geography',axis=1),geo_encoder_df],axis=1)



### Scale the input data
imput_data_scaled = scaler.transform(input_df) 

### Predict the churn
prediction = model.predict(imput_data_scaled)
prediction_proba = prediction[0][0]



st.write(f'Churn Probability: {prediction_proba:.2f}')
if prediction_proba  > 0.5:
    st.write('Customer is likely to churn')
else :
    st.write('Customer is not likely  churn')  