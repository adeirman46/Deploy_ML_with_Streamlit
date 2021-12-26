# Import necessary packages
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Import pipeline
pipe = pickle.load(open('penguin_pipeline.pkl', 'rb'))

# Add streamlit title and text
st.title('Penguin Machine Learning Classifier')
st.write('Classifying Penguin to 3 categories and train the model on the fly!')
st.subheader('Training Classification Model')

# Create penguin data loader
password_guess = st.text_input('What is the Password?')
if password_guess != st.secrets['password']:
    st.stop()
    
data = st.file_uploader(label='Upload your own penguin data')

@st.cache()
def load_data(data):
    if data is not None:
        df = pd.read_csv(data)
    else:
        st.stop()
    return df

df = load_data(data)

# Create number and categorical columns
num_cols = ['bill_length_mm', 'bill_depth_mm',
       'flipper_length_mm', 'body_mass_g', 'year']

cat_cols = ['island', 'sex']

# Input the test size
size = st.number_input(label='Set the test size',
                            min_value=0.1,
                            max_value=1.0)

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df[num_cols + cat_cols], df['species'], test_size=size, random_state=42)

# Transform the data
X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

# Fit the model
model_selection = st.selectbox(label='Choose your desired classification model algorithms', options=['K Nearest Neighbors', 'Random Forest', 'Decision Tree'])

if model_selection == 'K Nearest Neighbors':
    model = KNeighborsClassifier()
elif model_selection == 'Random Forest':
    model = RandomForestClassifier()
elif model_selection == 'Decision Tree':
    model = DecisionTreeClassifier()

st.write('Fitting the model...')
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
st.write(f'The training was finished wich accuracy {accuracy * 100:.2f}%')

# Add feature importances plot
st.subheader('Feature Importances')

sns.set_style('dark')
fig, ax = plt.subplots()
full_cols = list(df[num_cols].columns)
full_cols.extend(list(pd.get_dummies(df[cat_cols]).columns))
full_cols_copy = full_cols.copy()
ax = sns.barplot(np.sort(model.feature_importances_)[::-1], full_cols_copy, palette='viridis')
plt.tight_layout()
st.pyplot(fig)

# Add distribution plot of bill_length_mm
st.subheader('Distribution of Each Feature')

feature = st.selectbox(label='What feature you want to see the distributions?', options=df[num_cols].columns)
fig, ax = plt.subplots()
ax = sns.displot(x=feature, hue='species', data=df)
plt.axvline(df[feature].median(), color='red')
st.pyplot(ax)

# Set input
st.subheader('Creating Model Prediction')

with st.form('User Input'):
    bill_length = st.number_input(label='Bill Length (mm)',
                                  min_value=0.0)

    bill_depth = st.number_input(label='Bill Depth (mm)',
                                  min_value=0.0)

    flipper_length = st.number_input(label='Flipper Length (mm)',
                                     min_value=0.0)

    body_mass = st.number_input(label='Body Mass (g)',
                                min_value=0.0)

    year = st.number_input(label='Year',
                           min_value=0.0)

    island = st.selectbox(label="Island",
                          options=['Torgersen', 'Biscoe', 'Dream'])

    sex = st.selectbox(label='Gender', 
                       options=['female', 'male'])
    
    st.form_submit_button('Predict')

# Create dataframe to combine all of them
if year != 0:
    df = pd.DataFrame(data={'bill_length_mm':bill_length,
                            'bill_depth_mm':bill_depth,
                            'flipper_length_mm':flipper_length,
                            'body_mass_g':body_mass,
                            'year': year,
                            'island': island,
                            'sex': sex}, index=[0])
else:
    st.stop()

# Preprocess the data using pipeline
df = pipe.transform(df)

# Create predictions
st.subheader('Prediction')
pred = model.predict(df)

# Write and show the result
if pred[0] == 'Adelie':
    image_path = 'adelie.jpg'
elif pred[0] == 'Gentoo':
    image_path = 'gentoo.jpg'
elif pred[0] == 'Chinstrap':
    image_path = 'chinstrap.jpg'
st.image(image_path, width=360)  
st.write(f'The prediction says this penguin is {pred[0]} race')
