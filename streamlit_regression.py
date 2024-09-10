import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler , LabelEncoder
import pickle 


### load the model
model = tf.keras.models.load_model('regression_model.h5')

### Load the encoders and scaler 
with open('regression_label_encoder_gender.pkl','rb') as file:
    regression_label_encoder_gender = pickle.load(file)
with open('regresssion_onehot_encoder_geo.pkl','rb') as file:
    regresssion_onehot_encoder_geo = pickle.load(file)
with open('regression_scale.pkl','rb') as file:
    regression_scale = pickle.load(file)    
    
### streamlit app
st.title('Estimated Salary Prediction')    

## user input 
geography = st.selectbox('Geography',regresssion_onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',regression_label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure',0,10)
num_of_product = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_number = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame({
'CreditScore': [credit_score],	
'Geography': [geography],
'Gender': [regression_label_encoder_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_product],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_number],
'Exited': [exited]	    
       
}
    
    
)        




geo_encoder = regresssion_onehot_encoder_geo.transform(input_data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),columns =regresssion_onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df =pd.concat([input_data.drop('Geography',axis=1),geo_encoder_df],axis=1)



### Scale the input data
imput_data_scaled = regression_scale.transform(input_df) 

### Predict the churn
prediction = model.predict(imput_data_scaled)
prediction_proba = prediction[0][0]


st.write(f'Predicted Estimated Salary: ${prediction_proba:.2f}')