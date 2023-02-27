import streamlit as st
import pandas as pd
from datetime import datetime
import os

now = datetime.now()
date = now.strftime("%d/%m/%Y %H:%M:%S")

def format_download(pred):
    p=pd.Series(pred)
    t=pd.Series(st.session_state.y_test)
    yt = t.reset_index(drop=True)
    idx = pd.Series(t.index)
    formatted_data = pd.concat([idx,yt,p],axis=1,keys=['Ytest_Index','Ytest','Ypred'])
    return(formatted_data)

def download_file(pred,fname,model,task,accu_mse,ds):
    # Get the path to the user's download folder
    pred_df = format_download(pred)
    new_cols = pd.MultiIndex.from_tuples([('Dataset: '+str(ds),'Task: '+task,'Ytest_index'),('Model: '+model,'Date: '+str(date),'Ytest'),('Accuracy/MSE: '+str(accu_mse),' ','Ypred')])
    pred_df.columns = new_cols
    fname = fname+".csv"
    download_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    csv_path = os.path.join(download_folder, fname)
    pred_df.to_csv(csv_path, index=False)
