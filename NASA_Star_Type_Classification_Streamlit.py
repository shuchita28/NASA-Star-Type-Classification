#!pip install streamlit --quiet
import streamlit as st
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


stars_ohe = pd.read_csv('/Users/shuchitamishra/Desktop/Jobs/OA/NASA-Star-Type-Classification/FinalProcessed.csv')

scaler = StandardScaler()

stars_ohe_scaled = scaler.fit(stars_ohe.drop('Type', axis = 1))
stars_ohe_scaled = scaler.transform(stars_ohe.drop('Type', axis = 1))

X = pd.DataFrame(stars_ohe_scaled, columns = stars_ohe.columns[:-1])
Y = stars_ohe['Type']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42, test_size = 0.33)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.fit_transform(X_test)

model_knn_1 = KNeighborsClassifier(n_neighbors= 1)

model_knn_5 = KNeighborsClassifier(n_neighbors= 10)

model_knn_1.fit(X_train, Y_train)
model_knn_5.fit(X_train, Y_train)

predict_knn_1 = model_knn_1.predict(X_test)
predict_knn_5 = model_knn_5.predict(X_test)

#print(predict_knn_1)
#print(predict_knn_5)

#print(confusion_matrix(Y_test, predict_knn_1))
#print(confusion_matrix(Y_test, predict_knn_5))

print(classification_report(Y_test,predict_knn_1))
print(classification_report(Y_test,predict_knn_5))

#error_rate= []
#for i in range(1,40):
#    model_knn = KNeighborsClassifier(n_neighbors = i)
#    model_knn.fit(X_train,Y_train)
#    predict_knn_i = model_knn.predict(X_test)
#    error_rate.append(np.mean(predict_knn_i != Y_test))
#plt.figure(figsize = (10,6))
#plt.plot(range(1,40),error_rate,color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 3)
#plt.title('Error Rate vs K')
#plt.xlabel('K')
#plt.ylabel('Error Rate')

print(model_knn_5.score(X_test, Y_test))

cv_scores = cross_val_score(model_knn_1, X, Y, cv = 10, scoring = 'accuracy')
print(cv_scores)

avg_cv_scores = np.mean(cv_scores)
print(avg_cv_scores)

# choose k between 1 to 40
#k_range = range(1, 40)
#k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
#    k_scores.append(scores.mean())

# plot to see clearly
#plt.figure(figsize = (10,6))
#plt.plot(k_range, k_scores, color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 3)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross-Validated Accuracy')
#plt.show()

params={'n_neighbors': range(1,40)}
params

model_knn_grid = GridSearchCV(KNeighborsClassifier(), params, cv=10, scoring='accuracy')
model_knn_grid.fit(X_train, Y_train)

GridSearchCV(cv=10, estimator=KNeighborsClassifier(),
             param_grid={'n_neighbors': range(1, 40)}, scoring='accuracy')

print(model_knn_grid.best_estimator_)
print(model_knn_grid.best_score_)



st.header("NASA Star Type Classification:")
image = Image.open('/Users/shuchitamishra/Desktop/Jobs/OA/NASA-Star-Type-Classification/nasaimage.jpeg')
st.image(image, use_column_width=True)
st.write("Please insert values, to get Star type class prediction")

Temperature_ip = st.slider('Temperature:', -0.92 ,3.0)
L_ip = st.slider('Luminosity:', -0.6, 4.3)
R_ip = st.slider('Relative radius',-0.4, 3.5)
A_M_ip = st.slider('Absolute Magnitude:', -1.5, 3.5)
Type_ip = st.radio("Type:", ("-0.47" ,"2.12"))
Color_Blue_White_ip = st.radio("Is the color blue-white?", ["0", "1"])
Color_Orange_ip = st.radio("Is the color orange?", ["0", "1"])
Color_Red_ip = st.radio("Is the color red?", ["0", "1"])
Color_White_ip = st.radio("Is the color white?", ["0", "1"])
Color_White_Yellow_ip = st.radio("Is the color white-yellow?", ["0", "1"])
Class_B_ip = st.radio("Is the spectral class B?", ["0", "1"])
Class_F_ip = st.radio("Is the spectral class F?", ["0", "1"])
Class_G_ip = st.radio("Is the spectral class G?", ["0", "1"])
Class_K_ip = st.radio("Is the spectral class K?", ["0", "1"])
Class_M_ip = st.radio("Is the spectral class M?", ["0", "1"])
Class_O_ip = st.radio("Is the spectral class O?", ["0", "1"])

data = {'Temperature': Temperature_ip,
        'Luminosity': L_ip,
        'Relative radius': R_ip,
        'Absolute magnitude': A_M_ip,
        'Type' : Type_ip,
        'Color_Blue_White' : Color_Blue_White_ip,
        'Color_Orange' : Color_Orange_ip,
        'Color_Red' : Color_Red_ip,
        'Color_White' : Color_White_ip,
        'Color_White_Yellow' : Color_White_Yellow_ip,
        'Class_B' : Class_B_ip,
        'Class_F' : Class_F_ip,
        'Class_G' : Class_G_ip,
        'Class_K' : Class_K_ip,
        'Class_M' : Class_M_ip,
        'Class_O' : Class_O_ip
        }

features = pd.DataFrame(data, index=[0])
print(features.shape, X_train.shape)

pred_proba = model_knn_5.predict_proba(features)
#or
prediction = model_knn_5.predict(features)

st.subheader('Prediction Percentages:') 
st.write('**Probablity of Star Type being B is ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of Star Type being F is ( in % )**:',pred_proba[0][1]*100)
st.write('**Probablity of Star Type being G is ( in % )**:',pred_proba[0][2]*100)
st.write('**Probablity of Star Type being K is ( in % )**:',pred_proba[0][3]*100)
st.write('**Probablity of Star Type being M is ( in % )**:',pred_proba[0][4]*100)