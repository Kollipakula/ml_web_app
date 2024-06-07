# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
#loading the trained model
loaded_model=pickle.load(open('C:/Users/HP/OneDrive/Documents/Deploying_ML/trained_model.sav','rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array    Ensures that the input data is in a format compatible with machine learning models.
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped) #here instead of sv_model we used above trained model to save
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')