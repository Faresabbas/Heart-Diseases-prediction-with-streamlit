from http import HTTPStatus
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
# read the data set 
data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = np.array(data.target)

#Logistic Regression using  the data set 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X, y)

# app title
st.title("Heart Diseases")

# app sidebar
nav = st.sidebar.radio('options',["Home" , "Prediction","About Us"])

if nav == "Home" :
    st.image("heart.jpeg")
    st.markdown(""" ## Heart Disease Dataset""")
    st.markdown(""" ### About Dataset : """)
    st.markdown(""" #### Context : 
              " This data set dates from 1988 and consists of four databases:
               Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, 
               but all published experiments refer to using a subset of 14 of them. 
               The "target" field refers to the presence of heart disease in the patient. 
               It is integer valued 0 = no disease and 1 = disease." """)
    st.markdown(""" #### Content : """)
    st.text("""               1- age
               2- sex
               3- chest pain type (4 values)
               4- resting blood pressure
               5- serum cholestoral in mg/dl
               6- fasting blood sugar > 120 mg/dl
               7- resting electrocardiographic results (values 0,1,2)
               8- maximum heart rate achieved
               9- exercise induced angina
               10- oldpeak = ST depression induced by exercise relative to rest
               11- the slope of the peak exercise ST segment
               12- number of major vessels (0-4) colored by flourosopy
               13- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.""")
    
    st.dataframe( data)
if nav == "Prediction" :
    st.header("Heart Disease")
    age = st.slider("Age",18,90,step=1)
    sex = st.slider("sex  male = 0 , female = 1" , 0 ,1 , step=1)
    cp = st.slider("chest pain type (4 values)" , 0 ,3 , step=1)
    trestbps = st.slider("resting blood pressure" , 90 ,200 , step=1)
    chol = st.slider("serum cholestoral in mg/dl" , 125 ,565 , step=1)
    fbs = st.slider("fasting blood sugar > 120 mg/dl" , 0 ,1 , step=1)
    restecg = st.slider("resting electrocardiographic results (values 0,1,2)" , 0 ,2 , step=1)
    thalach = st.slider("maximum heart rate achieved" , 70 ,205 , step=1)
    exang = st.slider("exercise induced angina" , 0 ,1 , step=1)
    oldpeak = st.slider("oldpeak = ST depression induced by exercise relative to rest" , 0.00 ,6.20 , step=0.1)
    slope = st.slider("the slope of the peak exercise ST segment" , 0 ,2 , step=1)
    ca = st.slider("number of major vessels (0-4) colored by flourosopy" , 0 ,4 , step=1)
    thal = st.slider("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect The names and social security numbers of the patients" , 0 ,3 , step=1)
    
    val = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1)
    pred = clf.predict(val)[0]
    
    if st.button("predict") : 
        st.markdown("### It is integer valued 0 = no disease and 1 = disease.")
        st.success(f"the prediction : {(pred)}" )
        if pred == 0 : 
            st.write("You Are Ok ")
        else : 
            st.write("Ops Maybe you are sick")   
    
    st.title("Contribute")
    if st.checkbox("Do you want to add your data to our database ?! <to improve results only>")   :
        if st.button("Submit") :
            to_add = {"age":age , "sex":sex ,"cp":cp ,"trestbps":trestbps ,"chol":chol ,"fbs":fbs ,"restecg":restecg ,"thalach":thalach ,"exang":exang ,"oldpeak":oldpeak ,"slope":slope ,"ca":ca ,"thal":thal ,"target":pred }  
            # to_add = pd.DataFrame(to_add)
            # to_add =[age , sex ,cp ,trestbps ,chol ,fbs ,restecg ,thalach ,exang ,oldpeak ,slope ,ca ,thal ,pred]
            to_add = {k:[v] for k,v in to_add.items()}  # WORKAROUND
            to_add = pd.DataFrame(to_add)
            to_add.to_csv("heart.csv" ,mode="a",index=False , header=False)
            st.success("Added")
    
if nav == "About Us":
    st.header("Hi")
    st.markdown("### Welcome")
    st.write("My Name is "+"Fares Abbas" )
    st.markdown("### Contact me at whatsapp +201032902535")
    st.write("My kaggle : "+"www.kaggle.com/faresabbasai2022")
    st.write("my github : "+"https://github.com/Faresabbas") 
    st.write("my LinkedIn : "+"https://www.linkedin.com/in/fares-abbas/")
    st.text_input("Your feedback")
    st.header("Thanks for your feedback")   

        
