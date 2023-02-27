import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
##############################
import models
import file_download as dload
from data import DataSource
from metrics import Metrics
##############################
if 'webdata' not in st.session_state:
    st.session_state.webdata = 0  
if 'X_train' not in st.session_state:
    st.session_state.X_train = 0    
if 'y_train' not in st.session_state:
    st.session_state.y_train = 0
if 'X_test' not in st.session_state:
    st.session_state.X_test = 0
if 'y_test' not in st.session_state:
    st.session_state.y_test = 0       
if 'dataset_file' not in st.session_state:
    st.session_state.dataset_file=None

ds = DataSource()
met = Metrics()

tablist = ["**Data Source**","**Data Explorer**","**Classification**","**Regression**"]
t1,t2,t3,t4 = st.tabs(tablist)

st.sidebar.markdown('<span style="font-size:30px;font-family:Comic Sans MS;">Machine Learning</span>',unsafe_allow_html=True)
st.sidebar.write("by Fazal Masud Kundi")

# Data source
data_source = st.sidebar.selectbox(
    "**Data Source**",
    ("Local", "Web"),key=300)


# File header (sidebar)
file_header = st.sidebar.radio(
    "**File Header**",
    ("None", "Yes"),key=400)
if(file_header == 'Yes'):
    header = 0
else:
    header = None
    
# Image resize (sidebar)
image = Image.open("irisa.jpg")
new_image = image.resize((300, 220))
st.sidebar.image(new_image)

# Message display
def display_message(msg,color='black'):
    if(color=='red'):
        st.markdown(f'<span style="color:red;font-size:15px;font-family:Arial;">{msg}</span>',unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="font-size:15px;font-family:Arial;">{msg}</span>',unsafe_allow_html=True)

# Load data file
def load_data(file_name,ext,header,ds):
    if(ext.lower() in ['.csv','.data']):
        st.session_state.webdata = ds.csv(file_name,header)
        display_message("Data Loaded Successfully")
    elif(ext.lower() in ['.xls','.xlsx']):
        st.session_state.webdata = ds.excel(file_name,header)
        display_message("Data Loaded Successfully")
    else:
        st.write("Invalid data file")
        

    
# Data source (sidebar)
with t1:
    if(data_source=='Local'):
        
        display_message("Check for File header before loading",'red')
        upload = st.file_uploader("**Choose a file (Valid format: CSV (.csv,.data)and MS Excel Sheet (xls,.xlsx))**",
                                             type=["csv","data","xls","xlsx"],key=100)
        if(upload is not None):
            ext = os.path.splitext(upload.name)[1]
            st.session_state.dataset_file = upload.name
            load_data(upload,ext,header,ds) # calling function
        
    else:
        with st.form("web_form"):
            display_message("Check for File header before loading",'red')
            url = st.text_input("**Enter URL [Select option and press OK]**",value="",key=101,help="Only .CSV,.XSL or XSLX files")
            ok = st.form_submit_button("OK")
            ext = os.path.splitext(url)[1]
            if(ok):
                st.session_state.dataset_file = os.path.basename(url)
                load_data(url,ext,header,ds)
        

# Data Explorer   
with t2:
    # File data/statistics
    c21,c22 = st.columns(2)
    fs = c21.checkbox("**File Statistics**",key=500)
    fd = c22.checkbox("**File Data**",key=600)
    fc = c21.checkbox("**Correlation**",key=700)
    fp = c22.checkbox("**Plot**",key=800)
    c21.write("**Data Preprocessing**")
    c22.write(" ")
    c22.write(" ")
    mv = c21.checkbox("**Count Missing Values**",key=900,value=True)
    dmv = c22.checkbox("**Drop rows with missing values**",key="dmv_key")
    
    try:
        
        if(st.session_state.webdata is not None):
            if(dmv):
                n = st.session_state.webdata.isna().sum().sum()
                if(n>0):
                    st.session_state.webdata = st.session_state.webdata.dropna()
                    msg = f'{n} rows dropped'
                    display_message(msg,'red')
                    display_message("For undo uncheck drop option",'red')
                                
            if(fs):
                # File statistics
                st.write(st.session_state.webdata.describe())
            if(fd):
                # File data
                st.write(st.session_state.webdata)
            if(mv):
                st.write("Missing Values:",st.session_state.webdata.isna().sum().sum())
            if(fc):
                # Correlation
                st.write(st.session_state.webdata.corr())
            if(fp):
                # Boxplot
                fig, ax = plt.subplots()
                ax.boxplot(st.session_state.webdata.iloc[:,:-1])
                st.pyplot(fig)

                
    except Exception as e:
        display_message(e,'red')
        display_message("File not uploaded","red")
        
    
###################
# Classification  #
###################
    
with t3:
    clsf_list =["Classification Options","KNN","Decision Tree","SVM","Random Forest","Logistic Regression",
                "Naive Bayes"]
    t30,t31,t32,t33,t34,t35,t36 = st.tabs(clsf_list)


    ############################
    #process results
    def clsf_results(y_test,prediction,model,task):
        if(creport and prediction is not None):
            # met object of Metric class
            measures,accu = met.cmeasures(y_test,prediction)
            st.write(measures)
            
        if(cfm and prediction is not None):
            st.write("**Confusion Matrix**")
            cm = met.cfmatrix(y_test, prediction)
            st.write(cm)
        if(pred_screen):
            st.write(prediction)
        if(pred_file):
            dload.download_file(prediction,pred_file_name,model,task,accu,st.session_state.dataset_file)  # call download_file()
       
    
    # Train model and make prediction
    def train_and_predict(model,params):
        try:
            prediction = models.train(model,st.session_state.X_train,st.session_state.y_train,st.session_state.X_test,params)
            return(prediction)
        except Exception as exp:
            st.markdown(f'<span style="color:red;font-size:20px;font-family:Arial;">{exp}</span>',unsafe_allow_html=True)
            
            


        # Classification options
    with t30:
        
        co1,co2 = st.columns(2)
        co1.markdown('<span style="color:blue;font-size:20px;">Classification Options</span>',unsafe_allow_html=True)
        col301,col302,col303 = st.columns(3)
        tsize= col301.number_input("**Test Size (%age)**",value=20)
        test_size = tsize/100
        creport = col302.checkbox("**Classification Report**",value=False)        
        cfm = col303.checkbox("**Confusion Matrix**")  
        col302.write("")
        col302.write("")
        col301.write("**Prediction**")
        pred_file = col301.checkbox("**Print to File**",value=False)
        if(pred_file):
            pred_file_name = col301.text_input("**Enter file name without extension**",disabled=False)
        else:
            pred_file_name = col301.text_input("**Enter file name without extension**",disabled=True)
        
        pred_screen = col301.checkbox("**Print to Screen**",value=False)
        try:
            if(creport or cfm):
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = ds.split_data(st.session_state.webdata,test_size)
        except Exception as ex:
            display_message(ex,'red')
            display_message("Check classification options","red")
            display_message("File not uploaded","red")
            

    # KNN
    with t31:
        c311,c312,c313 = st.columns(3)
        c312.markdown('<span style="color:blue;font-size:20px;">KNN Classifier</span>',unsafe_allow_html=True)
        kn = c311.number_input("**Neighbour (k)**",min_value=1,value=5)
        params ={'n_neighbors':kn}
        if(st.button("Classify",key='fmkey1')):
            pred = train_and_predict('KNNC',params)
            clsf_results(st.session_state.y_test,pred,'KNN','Classification')
            
    # Decision tree
    with t32:
        c321,c322,c323 = st.columns(3)
        c322.markdown('<span style="color:blue;font-size:20px;">Decision Tree Classifier</span>',unsafe_allow_html=True)
        option = c321.selectbox('**Select split criterion**',('Gini', 'Entropy', 'Log_loss'),key='fmselckey1')
        params = {'criterion':option.lower()}
        if(st.button("Classify",key='fmkey2')):
            pred = train_and_predict('DTC',params)
            clsf_results(st.session_state.y_test,pred,'Decision Tree','Classification')
            
    # SVM
    with t33:
        c331,c332,c333 = st.columns(3)
        c332.markdown('<span style="color:blue;font-size:20px;">Support Vector Machine Classifier</span>',unsafe_allow_html=True)
        c = c331.number_input("**Regularization Parameter (C)**",min_value=0.1,value=1.0)
        kernel  = c331.selectbox('**Kernel type**',('RBF', 'Poly', 'Sigmoid','Linear'))
        dop = c331.number_input("**Degree of polynomial**",min_value=1,value=3)
        params = {'C':c,'kernel':kernel.lower(),'degree':dop}
        if(st.button("Classify",key='fmkey3')):
            pred = train_and_predict('SVMC',params)
            clsf_results(st.session_state.y_test,pred,'SVM','Classification')
            
    # Random Forest
    with t34:
        c341,c342,c343 = st.columns(3)
        c342.markdown('<span style="color:blue;font-size:20px;">Random Forest Classifier</span>',unsafe_allow_html=True)
        trees = c341.number_input("**Number of trees**",min_value=1,value=10)
        option = c341.selectbox('**Select split criterion**',('Gini', 'Entropy', 'Log_loss'),key='fmselckey2')
        params = {'n_estimators':trees,'criterion':option.lower()}
        if(st.button("Classify",key='fmkey4')):
            pred = train_and_predict('RFC',params)
            clsf_results(st.session_state.y_test,pred,'Random Forest','Classification')
            
    # Logistic Regression
    with t35:
        c351,c352,c353 = st.columns(3)
        c352.markdown('<span style="color:blue;font-size:20px;">Logistic Regression Classifier</span>',unsafe_allow_html=True)
        params = {'C':1.0}
        if(st.button("Classify",key='fmkey5')):
            pred = train_and_predict('LR',params)
            clsf_results(st.session_state.y_test,pred,'Logistic Regression','Classification')
            
    # Naive Bayes
    with t36:
        c361,c362,c363 = st.columns(3)
        c362.markdown('<span style="color:blue;font-size:20px;">Naive Bayes Classifier</span>',unsafe_allow_html=True)
        params = {'alpha':1.0}
        if(st.button("Classify",key='fmkey6')):
            pred = train_and_predict('NBC',params)
            clsf_results(st.session_state.y_test,pred,'Naive Bayes','Classification')

###############
# Regression  #
###############

with t4:
    tab_list = ["Regression Options","Linear","KNNRegressor","Decision Tree","SVM","Random Forest"]
    t40,t41,t42,t43,t44,t45 = st.tabs(tab_list)
    
    # Regression results
    def reg_results(yt,ypred,model,task):
        rres ={}
        if(errors):
            mse =mean_squared_error(yt,ypred)
            rmse = mean_squared_error(yt,ypred,squared=False)
            mae =  mean_absolute_error(yt,ypred)
            maxe = max_error(yt,ypred)
            rres['MSE'] = mse
            rres['RMSE'] = rmse
            rres['MAE'] = mae
            rres['Max_Error'] = maxe
        if(rsqrd):
            rsqr = r2_score(yt,ypred)
            rres['RSquared'] = rsqr

        rpdf = pd.DataFrame(rres,index=[''])
        st.write(rpdf)
        if(pred_screen):
            st.write(ypred)
        if(pred_file):
            dload.download_file(ypred,pred_file_name,model,task,rres['MSE'],st.session_state.dataset_file)
        
        
    # Regression options
    with t40:
        c401,c402,c403 = st.columns([1,2,1])
        c401.markdown('<span style="color:blue;font-size:20px;">Regression Options</span>',unsafe_allow_html=True)
        tsize= c401.number_input("**Test Size (%age)**",value=20,key="fmrtskry")
        test_size = tsize/100
        errors = c402.checkbox("**Errors(MSE/RMSE/MAE/MaxE)**",value=False)
        rsqrd = c403.checkbox("**R-Squared**",value=False)
        c401.write("**Prediction**")
        pred_file = c401.checkbox("**Print to File**",value=False,key="fmpfchk1")
        if(pred_file):
            pred_file_name = c401.text_input("**Enter file name**",disabled=False)
        else:
            pred_file_name = c401.text_input("**Enter file name**",disabled=True,key="fmpftin")
        
        pred_screen = c401.checkbox("**Print to Screen**",value=False,key="fmpscchk")

        
        try:
            if(errors or rsqrd):
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = ds.split_data(st.session_state.webdata,test_size)
        except Exception as ex:
            display_message(ex,"red")
            display_message("Check options","red")
            
    # Linear Regression
    with t41:
                        
        c411,c412,c413 = st.columns(3)
        c412.markdown('<span style="color:blue;font-size:20px;">Linear Regression</span>',unsafe_allow_html=True)
        params ={'fit_intercept':True}
        if(st.button("Compute",key='fmlrkey1')):
            pred = train_and_predict('LREG',params)
            reg_results(st.session_state.y_test,pred,'Linear','Regression')
            
    # KNN Regressor
    with t42:
        c421,c422,c423 = st.columns(3)
        c421.markdown('<span style="color:blue;font-size:20px;">KNN Regression</span>',unsafe_allow_html=True)
        kn = c421.number_input("**Enter neighbor (k)**",min_value=1,value = 5)
        params = {'n_neighbors':kn}
        if(st.button("Compute",key='fmknnkey1')):
            pred = train_and_predict('KNNR',params)
            reg_results(st.session_state.y_test,pred,'KNN','Regression')
            
    # Decision Tree Regressor
    with t43:
        c431,c432,c433 = st.columns(3)
        c431.markdown('<span style="color:blue;font-size:20px;">Decision Tree Regression</span>',unsafe_allow_html=True)
        option = c431.selectbox('**Select criterion**',("Squared_error","Friedman_mse","Absolute_error","Poisson"),key='fmselcdtkey1')
        params = {'criterion':option.lower()}
        if(st.button("Compute",key='fmdtrkey1')):
            pred = train_and_predict('DTR',params)
            reg_results(st.session_state.y_test,pred,'Decision tree','Regression')
            
    # SVM Regression (SVR)
    with t44:
        c441,c442,c443 = st.columns(3)
        c441.markdown('<span style="color:blue;font-size:20px;">SVM Regression</span>',unsafe_allow_html=True)
        c = c441.number_input("**Regularization Parameter (C)**",min_value=0.1,value=1.0,key="fmsvrkey")
        kernel  = c441.selectbox('**Kernel type**',('RBF', 'Poly', 'Sigmoid','Linear'),key="fmsbsvrkey")
        dop = c441.number_input("**Degree of polynomial**",min_value=1,value=3,key="fmipnsvrkey")
        params = {'C':c,'kernel':kernel.lower(),'degree':dop}
        if(st.button("Compute",key='fmsvrbtkey1')):
            pred = train_and_predict('SVMR',params)
            reg_results(st.session_state.y_test,pred,'SVM','Regression')
            
    # Random Forest Regression
    with t45:
        c451,c452,c453 = st.columns(3)
        c451.markdown('<span style="color:blue;font-size:20px;">Random Forest Regression</span>',unsafe_allow_html=True)
        trees = c451.number_input("**Number of trees**",min_value=1,value=10,key="fmrfionkey")
        option = c451.selectbox('**Select criterion**',("Squared_error","Friedman_mse","Absolute_error","Poisson"),key='fmserfkey2')
        params = {'n_estimators':trees,'criterion':option.lower()}
        if(st.button("Compute",key="fmrfbtkey")):
            pred = train_and_predict('RFR',params)
            reg_results(st.session_state.y_test,pred,'Random Forest','Regression')
            
            
