import streamlit as st
import pickle


with open("model.pkl","rb") as f:
    loaded=pickle.load(f)

model=loaded[0]
target_names=loaded[1]
sl_min,sl_max,sw_min,sw_max,pl_min,pl_max,pw_min,pw_max = loaded[2]

st.title("Iris Flower species prediction")

sl = st.slider("Sepal length(cm)",min_value=sl_min,max_value=sl_max,value=5.0,step=0.1)
pl = st.slider("Petal length(cm)",min_value=pl_min,max_value=pl_max,value=5.0,step=0.1)
sw = st.slider("Sepal Width(cm)",min_value=sw_min,max_value=sw_max,value=5.0,step=0.1)
pw = st.slider("Petal width(cm)",min_value=pw_min,max_value=pw_max,value=5.0,step=0.1)

if st.button('Predict'):
    input_data = [[sl,sw,pl,pw]]
    # prediction = model.predict(input_data)
    # prediction = 'Versicolour'
    prediction=model.predict(input_data)[0]

    st.write(f"The predicted species is : **{target_names[prediction]}**" )