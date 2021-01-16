import streamlit as st
import joblib


def encoder(var):
    if var == 'Yes' or var == 'Male':
        return 1
    else:
        return 0


def main():
    model = joblib.load('model.sav')
    polydipsia = st.radio(label="Have you recently observed a feeling of extreme thirstiness?", options=["Yes", "No"])
    polyuria = st.radio(label="Have you been using bathroom more frequently than before?", options=["Yes", "No"])
    weight_loss = st.radio(label="Have you observed a sudden weight loss lately?", options=["Yes", "No"])
    partial_paresis = st.radio(label="Have you observed partial loss of voluntary movement?", options=["Yes", "No"])
    Gender = st.radio(label="What is your gender?", options=["Male", "Female"])
    arr = [polydipsia, polyuria, weight_loss, partial_paresis, Gender]
    arr = list(map(encoder, arr))
    result = model.predict([arr])
    if st.button('Predict'):
        if result[0] == 1:
            st.error(
                'You have symptoms of early stage of diabetes-mellitus. Consult a qualified professional for further '
                'details.')
        else:
            st.success('You do not have any early stage symptoms of diabetes-mellitus.')


if __name__ == '__main__':
    st.title('Predict early stage diabetes-mellitus.')
    st.sidebar.markdown('Using a random forest classifier on UCI Early stage diabetes risk prediction dataset.')

    if not st.sidebar.button("About"):
        st.sidebar.text("kaustubh.shirpurkar@gmail.com ")
        st.sidebar.text("By Kaustubh Shirpurkar")
        st.sidebar.text("Built with Streamlit")

    main()
