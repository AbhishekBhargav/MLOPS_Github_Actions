import streamlit as st 

st.title("Iris Flower species prediction")

sl = st.slider("Sepal length(cm)",min_value=4.0,max_value=8.0,value=5.0,step=0.1)
pl = st.slider("Petal length(cm)",min_value=4.0,max_value=8.0,value=5.0,step=0.1)
sw = st.slider("Sepal Width(cm)",min_value=4.0,max_value=8.0,value=5.0,step=0.1)
pw = st.slider("Petal width(cm)",min_value=4.0,max_value=8.0,value=5.0,step=0.1)

if st.button('Predict'):
    input_data = [[sl,sw,pl,pw]]
    # prediction = model.predict(input_data)
    prediction = 'Versicolour'

    st.write(f"The predicted species is : **{prediction}**" )