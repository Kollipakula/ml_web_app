# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:09:52 2024

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('C:/Users/HP/OneDrive/Documents/Deploying_ML/trained_model.sav','rb'))


def diabetic_predective(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped) 
    print(prediction)
    if (prediction[0] == 0):
       return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    st.title('Diabetes Predection Web App')
    
    #getting input from the user
    
    
    Pregnancies=st.text_input('No of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('SkinThickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Value')
    Age=st.text_input('Age of the person')
    
    #Prediction
    diagnosis=''
    
    #Creating a button forprediction
    
    if st.button('Test Results'):
        diagnosis=diabetic_predective([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()