import streamlit as st

import pandas as pd 
from sklearn import datasets


#st.write("""# Iris Flower Prediction App""" )
st.title('Iris Classification App')
'Please adjust the input parameters in the left side menu bar  '
'ML model will auto run in the background & automatically classifies the category of Iris'

st.sidebar.header('User Input Parameters')

def user_input_features():

	sl = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
	sw = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
	pl = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
	pw = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

	data = {'sl':sl, 
				'sw':sw, 
				'pl':pl, 
				'pw':pw}

	features = pd.DataFrame(data, index=[0])
	return features

df = user_input_features() #side menu appears

st.subheader('User Input')
st.write(df)

#Images
pic_file = open('iris1.jpg','rb')
pic = pic_file.read()

pic_file0 = open('iris0.jpeg','rb')
pic0 = pic_file0.read()

pic_file2 = open('iris2.jpg','rb')
pic2 = pic_file2.read()

pic_list = [pic0,pic,pic2]

# Import model
import joblib
model = joblib.load('iris_model')

pred = model.predict(df)
pred_proba = model.predict_proba(df)

iris = datasets.load_iris()

print(type(pred)) 
import numpy as np
p = np.asscalar(pred) #As pred is a ndarray, we convert it into scalar & store it in p
print(type(p))


st.subheader('Prediction')
'Numerical Value : ', p #like print # magic commands
'Category : ',iris.target_names[p] 
st.image(pic_list[p], caption='Iris', use_column_width=True)



st.subheader('Probability')
pred_proba


st.write('> > _Reference :_')
iris.target_names

#Insert image

#st.image(pic, caption='Iris flower', use_column_width=True)

st.write('## Created by Sri Lakshmi') #add website, twitter link at contact



