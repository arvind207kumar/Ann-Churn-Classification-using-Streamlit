# Customer Churn Classification with ANN and Streamlit 🎉

# 🏆 Data Features:

RowNumber, CustomerId, Surname: Essential for keeping track but not impacting churn prediction.
CreditScore: Higher scores often mean lower risk.
Geography: Encoded as France, Germany, Spain.
Gender: Encoded to detect patterns across demographics.
Age: Older customers may be more loyal—or not!
Tenure: The longer the tenure, the higher the loyalty, usually.
Balance, EstimatedSalary: Higher figures don’t always mean a customer will stay.
NumOfProducts, HasCrCard, IsActiveMember: Key activity indicators.
Exited: The all-important target column, showing if a customer churned.

# Project Goal 🚀
Ever wondered why some customers stay loyal while others walk away? This project aims to predict customer churn using Artificial Neural Networks (ANN), helping businesses identify customers at risk of leaving. By analyzing essential features like Credit Score, Geography, Age, and more, we build a model that provides valuable insights for strategic decision-making. And the best part? You can interact with it directly through a Streamlit app that brings AI predictions to life in a super-friendly interface!
## Project Development 📈
Data Preparation & Encoding Magic 🧙‍♂️
We start by preprocessing the dataset, which includes features like CreditScore, Geography, Gender, Age, Balance, EstimatedSalary, and whether or not the customer Exited. To make sure our ANN can understand categorical data:

Label Encoding is used for Gender.
One-Hot Encoding brings out the magic in Geography (France, Germany, Spain), allowing each region to have its own column.
These transformations are saved for future use so the model can recognize new data input just as smoothly.
Building the ANN with TensorFlow 🧠
Using TensorFlow, we crafted a custom ANN with:

Input Layer for our features.
Hidden Layers that reveal patterns in customer behavior.
Output Layer to predict whether a customer will stay or go.
Hyperparameter tuning for the best combination of learning rate, batch size, number of epochs, and layer units makes the model more accurate with each training cycle.
Saving Weights & Encodings 💾
After training, we store the model’s weights as a .h5 file, so it’s ready for quick predictions without retraining. Plus, we save the encoding weights of Gender and Geography to ensure the model can understand new inputs post-training!

## Streamlit App Magic 🪄
Now comes the fun part: Streamlit! Our app offers a simple, interactive interface where you can:

Input customer data and receive instant predictions on whether they'll churn.
Adjust hyperparameters and observe the impact on model performance.
Load your own data files for real-time churn analysis, visualized directly in the app.
# 🌟 Key Features
Hyperparameter Tuning On-the-Fly: Adjust settings like batch size, epochs, and learning rate in the Streamlit app, instantly seeing how they impact model accuracy.
Weights Saved for Speedy Inference: The .h5 weights file means fast predictions without retraining, while label and one-hot encoding transformations ensure new data is seamlessly processed.
User-Friendly Interface: The Streamlit app puts the power in your hands with sliders, drop-downs, and charts to visualize model behavior.
Comprehensive Customer Insights: By analyzing features like Age, Balance, NumOfProducts, and IsActiveMember, the model identifies patterns to predict loyalty.

# Future Ideas 💡
Additional Visualizations: Graphs and analytics on customer features could provide further insights.
Model Expansion: Experiment with more advanced deep learning models or ensemble approaches.
API Integration: For seamless deployment, integrate the model with a live API for real-time churn prediction.
