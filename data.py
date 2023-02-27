import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSource:
    def display_error(self,err):
        st.markdown(f'<span style="color:red;font-size:15px;font-family:Arial;">{err}</span>',unsafe_allow_html=True)
    # Web data file    
    def csv(self,url,header=None):
        try:
            filedata = pd.read_csv(url,header=header)
            return(filedata)
        except Exception as exp:
            self.display_error(exp)
            

    def excel(self,url,header=None):
        try:
            filedata = pd.read_excel(url,header=header)
            return(filedata)
        except Exception as exp:
            self.display_error(exp)

    def split_data(self,fdata,test_size):
        try:
            X = fdata.iloc[:,:-1]
            y = fdata.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            return(X_train, X_test, y_train, y_test)
        except Exception as exp:
            self.display_error(exp)
            
        
