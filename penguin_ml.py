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
from streamlit_lottie import st_lottie
import requests
from streamlit_embedcode import github_gist
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Set streamlit page config
st.set_page_config(layout='wide')

# Import pipeline
pipe = pickle.load(open('penguin_pipeline.pkl', 'rb'))

# Add streamlit title and text
st.title('Penguin Machine Learning Classifier')
st.write('Classifying Penguin to 3 categories and train the model on the fly!')
st.subheader('Training Classification Model')
# Create penguin data loader
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

# Add side bar subheading
st.sidebar.header('Adjust the hyperparameter here!')

# Input the test size
size = st.sidebar.number_input(label='Set the test size',
                            min_value=0.1,
                            max_value=1.0)

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df[num_cols + cat_cols], df['species'], test_size=size, random_state=42)

# Transform the data
X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

# Select the model on sidebar
model_selection = st.sidebar.selectbox(label='Choose classification model algorithms', options=['K Nearest Neighbors', 'Random Forest', 'Decision Tree'])

if model_selection == 'K Nearest Neighbors':
    model = KNeighborsClassifier()
elif model_selection == 'Random Forest':
    model = RandomForestClassifier()
elif model_selection == 'Decision Tree':
    model = DecisionTreeClassifier()

# Fit the model
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.sidebar.success(f'The training was finished with accuracy {accuracy * 100:.2f}%')
  
# Split the graph 
cols_1, cols_2 = st.columns(2)

with cols_1:
    # Add distribution plot of bill_length_mm
    st.subheader('Distribution of Each Feature')
    
    # Select feature that want to be plotted
    feature = st.selectbox(label='What feature you want to see the distributions?', options=df[num_cols].columns)

    fig, ax = plt.subplots()
    ax = sns.displot(x=feature, hue='species', data=df)
    plt.axvline(df[feature].median(), color='red')
    st.pyplot(ax)
    
with cols_2:
    # Add scatter plot
    st.subheader('Correlations Between Two Features')
    
    # Select feature that want to be plotted
    option_1 = st.selectbox(label='Feature 1', options=df[num_cols].columns)
    option_2 = st.selectbox(label='Feature 2', options=df[num_cols].columns)
    
    fig, ax = plt.subplots()     
    ax = sns.scatterplot(x=option_1, y=option_2, data=df, hue='species', style='species')
    st.pyplot(fig)

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
    
    submitted = st.form_submit_button('Predict')
    
    if submitted:
        # Create dataframe to combine all of them
        submitted_data = pd.DataFrame(data={'bill_length_mm':bill_length,
                                            'bill_depth_mm':bill_depth,
                                        'flipper_length_mm':flipper_length,
                                            'body_mass_g':body_mass,
                                            'year': year,
                                            'island': island,
                                            'sex': sex}, index=[0])
        
        # Preprocess the data using pipeline
        submitted_data = pipe.transform(submitted_data)

        # Create predictions
        st.subheader('Prediction')
        pred = model.predict(submitted_data)
        proba = model.predict_proba(submitted_data)

        # Write and show the result
        # Load lottie
        def load_lottie_url(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        lottie_penguin = load_lottie_url('https://assets2.lottiefiles.com/packages/lf20_1cgsfbmb.json')
        st_lottie(lottie_penguin, height=350)
 
        st.write(f'The prediction says this penguin is {pred[0]} race with probability {proba.max() * 100} %')

# Add EDA plot
st.subheader("Pandas Profiling on Palmer's Penguin")
penguin_profile = ProfileReport(df, explorative=True)
st_profile_report(penguin_profile) 

# Add code
st.subheader('The Code')
st.write('See the following code that this website is built')
github_gist('https://gist.github.com/adeirman46/7d8480b22d7c6dbb78bba168bf5e10f5')
