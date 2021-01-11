import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def encoder(var):
    if var == 'Yes' or var == 'Male':
        return 1
    else:
        return 0

def main():
    model = joblib.load( 'model.sav' )
    df = pd.read_csv('top_features.csv')
    Polydipsia = st.radio( label="Have you recently observed a feeling of extreme thirstiness?", options=["Yes", "No"])
    Polydipsia = encoder(Polydipsia)
    Polyuria = st.radio( label="Have you been using bathroom more frequently than before?", options=["Yes", "No"] )
    Polyuria = encoder(Polyuria)
    weight_loss = st.radio( label="Have you observed a sudden weight loss lately?", options=["Yes", "No"] )
    weight_loss = encoder(weight_loss)
    partial_paresis = st.radio( label="Have you observed partial loss of voluntary movement?", options=["Yes", "No"] )
    partial_paresis = encoder(partial_paresis)
    Gender = st.radio( label="What is your gender?", options=["Male", "Female"] )
    Gender = encoder(Gender)
    arr = [Polydipsia, Polyuria, weight_loss, partial_paresis, Gender]
    result = model.predict([arr])
    if st.button('Predict'):
        if result[0] == 1:
            st.error(
                'You have symptoms of early stage of diabetes-mellitus. Consult a qualified professional for further '
                'details.' )
        else:
            st.success( 'You do not have any early stage symptoms of diabetes-mellitus.' )

if __name__ == '__main__':
    st.title('Predict early stage diabetes-mellitus.')
    st.sidebar.markdown('Using a random forest classifier on UCI Early stage diabetes risk prediction dataset.')

    if st.sidebar.button("About") == False:
        st.sidebar.text( "kaustubh.shirpurkar@gmail.com " )
        st.sidebar.text( "By Kaustubh Shirpurkar" )
        st.sidebar.text( "Built with Streamlit" )
    main()