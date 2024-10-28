# Customer Churn Classification with ANN and Streamlit 🎉

# Project Goal 🚀
Ever wondered why customers say goodbye to a service? Our goal was to predict customer churn — that moment when users drift away. This project uses Artificial Neural Networks (ANN) to dive into customer behavior, aiming to classify whether they’ll stay or leave, helping companies keep loyal customers around.

With Streamlit, our model is wrapped up in a user-friendly app that even non-technical folks can use to try out custom parameters and see the model’s predictions in action!

# Project Development 📈
Data Wrangling Magic 🎩
Data cleanup comes first! For categorical features like Gender, we applied label encoding, transforming it into machine-readable format (stored for later in .pkl files). For geographical data (think Spain, France, and Germany), we performed one-hot encoding to let the model understand their unique value, also stored for easy reuse.

## Architecting the ANN 🔍
We built an Artificial Neural Network using TensorFlow, designed to predict customer churn based on various features:

Input Layer: Customer features including age, balance, location, etc.
Hidden Layers: Multiple layers with activations to learn deep patterns within the data.
Output Layer: The result — will they stay (0) or leave (1)? The model makes it happen!
Hyperparameter Tuning 🎛️
With Streamlit as our hyperparameter playground, users can adjust key model parameters like the number of hidden layers, neurons, and learning rate. The app displays how tweaks affect model performance, making tuning interactive and insightful.

# Model Saving 🗄️
After training, the model’s weights are saved in .h5 format, ensuring that our carefully tuned ANN can be loaded, adjusted, and improved later without re-training from scratch. Same goes for the categorical encodings — everything is safely stored for reproducibility.

## Streamlit App Development 🖥️
We wrapped up the entire workflow in a sleek Streamlit app, where users can explore data preprocessing, test different hyperparameters, and view real-time prediction results. Think of it as a custom-built control panel for exploring churn analysis.

# Key Features ✨
Interactive Hyperparameter Tuning: Adjust the model's settings on the fly and see the effect on training, validation accuracy, and loss.
Stored Model and Encoding Weights: Avoid retraining by storing model weights and label encodings, keeping data consistent and reusable.
User-Friendly Dashboard: The Streamlit app is packed with visuals, real-time prediction capabilities, and metrics to explain model behavior clearly.
Business-Driven Insights: By identifying at-risk customers, this model can help businesses reduce churn and retain loyal users.

# Future Ideas 💡
Additional Visualizations: Graphs and analytics on customer features could provide further insights.
Model Expansion: Experiment with more advanced deep learning models or ensemble approaches.
API Integration: For seamless deployment, integrate the model with a live API for real-time churn prediction.
