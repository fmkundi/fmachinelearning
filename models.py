import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR

models = {}
models['LREG'] = LinearRegression()
models['LR'] = LogisticRegression()
models['KNNR'] = KNeighborsRegressor()
models['KNNC'] = KNeighborsClassifier()
models['DTC'] = DecisionTreeClassifier()
models['DTR'] = DecisionTreeRegressor()
models['NBC'] = MultinomialNB()
models['NBR'] = GaussianNB()
models['SVMC'] = SVC()
models['SVMR'] = SVR()
models['RFC'] = RandomForestClassifier()
models['RFR'] = RandomForestRegressor()


def train(model,XT,yt,XTest,params):
    try:
        
        clsf = models[model]
        clsf.set_params(**params)
        clsf = clsf.fit(XT,yt)
        pred = clsf.predict(XTest)
        return(pred)
    except Exception as ex:
        st.markdown(f'<span style="color:red;font-size:20px;">Select valid options:{ex}</span>',unsafe_allow_html=True)
        
    
    

