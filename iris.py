import streamlit as st
import pandas as pd 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
print(dir(iris))
print(iris.feature_names)
print(iris.target_names)
print(type(iris.target_names))

x = iris.data
y = iris.target 

# To check which model to use, we split & train &  test

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.2, random_state=1)

from sklearn.svm import SVC 
'''
model = SVC()
model.fit(x_tr,y_tr)
pred1 = model.predict(x_te)
print(model.score(x_te,y_te))
print(f1_score(y_te,pred1,average='weighted'))'''

clf= RandomForestClassifier()
clf.fit(x_tr,y_tr)
pred2 = clf.predict(x_te)
print(clf.score(x_te,y_te))

from sklearn.metrics import f1_score
print(f1_score(y_te,pred2,average='weighted'))

# Lets train on whole iris dataset & create a final model
model = RandomForestClassifier()
model.fit(x,y)


# Save model to use later in another file
import joblib
joblib.dump(model, 'iris_model')

print(x_te[:5])

'''
[[5.8 4.  1.2 0.2]
 [5.1 2.5 3.  1.1]
 [6.6 3.  4.4 1.4]
 [5.4 3.9 1.3 0.4]
 [7.9 3.8 6.4 2. ]]

'''

"""
st.write("""
# Iris Flower Prediction App
"""
This app predicts **Iris Flower** type
"""
"""
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
st.write(df)"""
a = [10,20,30]
st.write(iris.target_names)
st.write(a)