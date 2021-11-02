import streamlit as st
import numpy as np
import pandas as pd
import joblib
import nltk
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from PIL import Image


image=Image.open("images/logo.png")
st.image(image,width=80)



st.title('Goblin Ghost Ghoul Predictor')

with st.form(key='tweet_form'):
    bone_length = st.slider("Bone Length(Normalized): ", min_value=0.05,   
                       max_value=0.85, value=0.72, step=0.01)
    rottling_flesh = st.slider("Percentage of Rottling Flesh: ", min_value=0.05,   
                       max_value=1.24, value=0.52, step=0.01)
    hair_length = st.slider("Hair Length(Normalized): ", min_value=0.1,   
                       max_value=1.0, value=0.4, step=0.05)
    has_soul = st.slider("Percentage Of Soul: ", min_value=0.009,   
                       max_value=1.05, value=0.25, step=0.01)

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    
    clf=joblib.load("models/clf.joblib")

    y=clf.predict([[bone_length,rottling_flesh,hair_length,has_soul]])

    st.write(f"Class : {y}")
    
    if y=="Ghost":
        image_ghost = Image.open('images/ghost.png')
        st.image(image_ghost,caption='It is Ghost',width=700)

    elif y=="Goblin":
    
        image_goblin = Image.open('images/goblin.jpg')
        st.image(image_goblin,caption='It is Goblin')
    else:
        image_ghoul = Image.open('images/ghoul.jpeg')
        st.image(image_ghoul,caption='It is Ghoul')

    


