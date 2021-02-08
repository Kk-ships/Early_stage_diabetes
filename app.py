import streamlit as st
import joblib
import pandas as pd
from train import svc, rf, KNN, adaboost, gboost, extra_tree


def write_prediction(model, df):
    prediction = model.predict(df)[0]
    if prediction == 'Positive':
        st.error('You have high probability of having early symptoms for diabetes. Consult a specialist '
                 'immediately.')
    else:
        st.success("You have a very low probability of having early signs of diabetes.")


def main():
    model = st.selectbox(label='Choose machine learning model to use?',
                         options=[svc, rf, KNN, adaboost, gboost, extra_tree])
    polydipsia = st.radio(label="Have you recently observed a feeling of extreme thirstiness?", options=["Yes", "No"])
    polyuria = st.radio(label="Have you been using bathroom more frequently than before?", options=["Yes", "No"])
    weight_loss = st.radio(label="Have you observed a sudden weight loss lately?", options=["Yes", "No"])
    partial_paresis = st.radio(label="Have you observed partial loss of voluntary movement?", options=["Yes", "No"])
    Gender = st.radio(label="What is your gender?", options=["Male", "Female"])
    Age = st.slider(label='What is your age?', min_value=10, max_value=100, value=35)
    df = pd.DataFrame(data={'Polydipsia': [polydipsia],
                            'Polyuria': [polyuria],
                            'sudden weight loss': [weight_loss],
                            'partial paresis': [partial_paresis],
                            'Gender': [Gender], 'Age': [Age]})
    if model == rf:
        pipeline_rf = joblib.load('model_random_forest.sav')
        if st.button('Predict'):
            write_prediction(pipeline_rf, df)
    elif model == KNN:
        pipeline_knn = joblib.load('model_random_forest.sav')
        if st.button('Predict'):
            write_prediction(pipeline_knn, df)
    elif model == svc:
        pipeline_svc = joblib.load('model_SVC.sav')
        if st.button('Predict'):
            write_prediction(pipeline_svc, df)
    elif model == adaboost:
        pipeline_ada = joblib.load('model_adaboost.sav')
        if st.button('Predict'):
            write_prediction(pipeline_ada, df)
    elif model == gboost:
        pipeline_gbc = joblib.load('model_gradient_boosting.sav')
        if st.button('Predict'):
            write_prediction(pipeline_gbc, df)
    elif model == extra_tree:
        pipeline_etc = joblib.load('model_extra_tree.sav')
        if st.button('Predict'):
            write_prediction(pipeline_etc, df)


if __name__ == '__main__':
    st.title('Predict early stage diabetes-mellitus.')
    st.sidebar.markdown('Using a random forest/KNN/SVC/Adaboost/Gradient boosting classifier/Extra tree classifier'
                        ' on UCI Early stage diabetes risk prediction dataset.')

    if not st.sidebar.button("About"):
        st.sidebar.text("kaustubh.shirpurkar@gmail.com ")
        st.sidebar.text("By Kaustubh Shirpurkar")
        st.sidebar.text("Built with Streamlit")

    main()
