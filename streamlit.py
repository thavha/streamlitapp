import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
import pandas as pd

import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

import warnings
warnings.filterwarnings("ignore")


model = pickle.load(open('dt_predections.pkl', 'rb'))
encoder_dict = pickle.load(open('dt_predections.pkl', 'rb')) 


def main(): 
    raw = pd.read_csv("train.csv")
    st.title("Twitter sentiment analysis")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.subheader("Climate change tweet classification")
    st.markdown(html_temp, unsafe_allow_html = True)

    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.info("General Information")
	
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page


    if selection == "Prediction":
        st.info("Prediction with ML Models")
		# Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classify"):
            data = {'message': tweet_text}
      
            df=pd.DataFrame([list(data.values())], columns=['message'])
            le = preprocessing.LabelEncoder()
            le.classes_ = encoder_dict['message']
            df['message'] = le.transform(df['message'])
            features_list = df.values.tolist()      
            prediction = model.predict(features_list)

    
            output = prediction[0]

            st.success(output)

if __name__=='__main__': 
    main()
