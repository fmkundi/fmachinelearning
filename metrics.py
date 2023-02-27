import streamlit as st
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
import pandas as pd

class Metrics:
    
    # Display classification report
    def display_report(self,creport,accuracy):
        lines = creport.split("\n")
        index =[]
        report_data =[]
        rows = len(lines[2:-4])
        r=1
        for line in lines[2:-4]:
            row_data={}
            row = line.strip().split()
            if(len(row)>0):
                r = r + 1
                index.append(row[0])
                row_data['Class'] = row[0]
                row_data['Precision'] = row[1]
                row_data['Recall'] = row[2]
                row_data['F1_Score'] = row[3]
                row_data['Support'] = row[4]
                if(r<rows):
                    row_data['Accuracy']=' '
                else:
                    row_data['Accuracy']=accuracy
                report_data.append(row_data)
        return(pd.DataFrame(report_data))

        
    def cmeasures(self,y_test,pred):
        accuracy = accuracy_score(y_test,pred)
        clsrep = classification_report(y_test,pred)
        clsrepdf= self.display_report(clsrep,accuracy)
        return(clsrepdf,accuracy)

    def cfmatrix(self,y_test,pred):
        cfm = confusion_matrix(y_test, pred)
        return(cfm)
                
