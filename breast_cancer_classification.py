import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io 
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc


data=pd.read_csv('C:\\Users\\janha\\OneDrive\\Desktop\\ML\\data.csv')
label_encoder=LabelEncoder()
data['diagnosis']=label_encoder.fit_transform(data['diagnosis'])
cancer_data = data[['diagnosis', 'radius_mean', 'area_mean','compactness_mean','texture_mean', 'smoothness_mean','symmetry_mean','area_se', 'fractal_dimension_se','perimeter_worst', 'area_worst', 'compactness_worst','concavity_worst', 'concave points_worst', 'texture_worst', 'smoothness_worst']]
X = cancer_data.drop("diagnosis", axis=1)
Y = cancer_data['diagnosis']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    

def plt_metrics(mlist):

    if 'Confusion Matrix' in mlist:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(Y_test, y_pred)
        fig,ax = plt.subplots()
        sns.heatmap(cm, annot = True)
        st.pyplot(fig)
            
    if 'Precision-Recall Curve' in mlist:
        st.subheader('Precision-Recall Curve')        
        precision, recall, thresholds = precision_recall_curve(Y_test, y_pred)
        fig = px.area(x=recall, y=precision,title=f'Precision-Recall Curve (AUC={auc(precision, recall):.4f})',labels=dict(x='Recall', y='Precision'),width=700, height=500)
        fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=1, y1=0)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        st.write(fig)

    if 'ROC Curve' in mlist:
        
        from sklearn.metrics import roc_curve

        fpr1, tpr1, thresh1 = roc_curve(Y_test, pred_prob[:,1], pos_label=1)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)
        
        from sklearn.metrics import roc_auc_score
        fig,ax = plt.subplots()
        auc_score1 = roc_auc_score(Y_test, pred_prob[:,1])
        plt.plot(fpr1, tpr1, linestyle='--',color='orange')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        st.pyplot(fig)




#menubar
menu=st.sidebar.radio("Menu",['About Breast Cancer','About Project','Models Evaluation','Breast Cancer Prediction'])

#About_Breast_Cancer
if menu=='About Breast Cancer':

    st.title("Breast Cancer :")
    st.write("  ")
    
    col1, col2= st.columns(2)
    with col1:
        st.write("Breast cancer originates in your breast tissue.It occurs when breast cells mutate (change) and grow out of control, creating a mass of tissue (tumor). Like other cancers, breast cancer can invade and grow into the tissue surrounding your breast. It can also travel to other parts of your body and form new tumors. ")
        st.write("Breast cancer is one of the most common cancers among women, second only to skin cancer. It’s most likely to affect women over the age of 50.Though rare, men can also develop breast cancer. ")
    with col2:
        image = Image.open('C:\\Users\\janha\\OneDrive\\Desktop\\ML\\download.jpeg')
        st.image(image)
    st.write("  ")
    st.header("**Breast Tumors :**")
    st.write("There are two types of breast cancer tumors: those that are non-cancerous, or ‘benign’, and those that are cancerous, which are ‘malignant’.")
    st.write("  ")
    st.write("**Benign Tumors :**")
    st.write("When a tumor is diagnosed as benign, doctors will usually leave it alone rather than remove it.If they grow inducing pain these tumors are removed.")
    st.write("**Malignant Tumors :**")
    st.write("Malignant tumors are cancerous and may be aggressive because they invade and damage surrounding tissue.When a tumor is suspected to be malignant, the doctor will perform a biopsy to determine the severity or aggressiveness of the tumor.")

    st.write("  ")
    st.header("Types of Breast Cancer :")
    st.markdown("* Ductal Carcinoma In Situ (DCIS)")
    st.markdown("* Invasive Ductal Carcinoma (IDC)")
    st.markdown("* Lobular Carcinoma In Situ (LCIS)")
    st.markdown("* Invasive Lobular Cancer (ILC)")
    st.markdown("* Triple Negative Breast Cancer")
    st.markdown("* Inflammatory Breast Cancer (IBC)")
    st.markdown("* Metastatic Breast Cancer")
    st.markdown("* Breast Cancer During Pregnancy")

    st.write("  ")
    st.header("Causes of Breast Cancer :")
    st.markdown("* Agebeing 55 or older.")
    st.markdown("* Women more likely to have breast cancer.")
    st.markdown("* Family History and Genetics")
    st.markdown("* Smoking and Alcohol Consumption")
    st.markdown("* Obesity")
    st.markdown("* Hormone Replacement Therapy")
    
    st.write("  ")
    st.header("Breast Cancer Treatments :")
    st.markdown("* Breast cancer surgery")
    st.markdown("* Chemotherapy for breast cancer")
    st.markdown("* Radiation therapy for breast cancer")
    st.markdown("* Hormone therapy for breast cancer")
    st.markdown("* Immunotherapy for breast cancer")
    st.markdown("* Targeted drug therapy for breast cancer")
    
    
#About_Project
if menu=='About Project':
   
    data=pd.read_csv('C:\\Users\\janha\\OneDrive\\Desktop\\ML\\data.csv')
    label_encoder=LabelEncoder()
    data['diagnosis']=label_encoder.fit_transform(data['diagnosis'])
    cancer_data = data[['diagnosis', 'radius_mean', 'area_mean','compactness_mean','texture_mean', 'smoothness_mean','symmetry_mean','area_se', 'fractal_dimension_se','perimeter_worst', 'area_worst', 'compactness_worst','concavity_worst', 'concave points_worst', 'texture_worst', 'smoothness_worst']]
    X = cancer_data.drop("diagnosis", axis=1)
    Y = cancer_data['diagnosis']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    data_mean = data.iloc[:, 2:11]
    data_worst = data.iloc[:, 22:-1]
    data_se = data.iloc[:, 11:21]
    
    st.title('Breast Cancer Classification')
    image = Image.open('C:\\Users\\janha\\OneDrive\\Desktop\\ML\\breast_cancer.png')
    st.image(image)
    st.write()
    data_info=st.selectbox("About Data :",['Dataset Information','Tabular Data','Features','Statistical summary of Data'])  
    if data_info=='Dataset Information':
        st.write()
        st.subheader("Dataset Information :")
        st.write('''The dataset used  is Breast Cancer
        Wisconsin (Diagnostic) Data Set.Dataset is downloaded from Kaggle. Features are computed from a digitized
        image of a fine needle aspirate (FNA) of a breast mass. They describe
        characteristics of the cell nuclei present in the image.

        Attribute Information:

        1) ID number
        2) Diagnosis (M = malignant, B = benign)
        (3-32)
        Ten real-valued features are computed for each cell nucleus:
        
        a) radius (mean of distances from center to points on the perimeter)
        b) texture (standard deviation of gray-scale values)
        c) perimeter
        d) area
        e) smoothness (local variation in radius lengths)
        f) compactness (perimeter^2 / area - 1.0)
        g) concavity (severity of concave portions of the contour)
        h) concave points (number of concave portions of the contour)
        i) symmetry
        j) fractal dimension (coastline approximation - 1)

        The mean, standard error and worst or largest (mean of the three
        largest values) of these features were computed for each image,
        resulting in 30 features.''')

    if data_info=='Tabular Data':
        st.subheader('Top 10 records :')
        st.table(data.head(10))

    if data_info=='Features':
        st.subheader("Features Information :")
        info=data.info()
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
    if data_info=='Statistical summary of Data':
        st.subheader('Statistical summary')
        st.table(data.describe())
        
    st.write()    
    st.title("Graphs :")
    graphs=st.selectbox("Select any one graph",['Correaltion Heatmap','Total Number of Malignant and Benign','Histogram for Mean Values','Histogram for Standard Error Values','Histogram for Worst Values'])  

    if graphs=='Correaltion Heatmap':
        st.header("Correaltion Heatmap :")
        correlation = cancer_data.corr() 
        fig,ax=plt.subplots(figsize=(18, 12))
        sns.heatmap(correlation, cmap='rocket', annot=True)
        st.pyplot(fig)
        
    if  graphs=='Total Number of Malignant and Benign':
        st.header("Total Number of Malignant and Benign")
        fig2,ax=plt.subplots(figsize=(18, 12))
        graph=sns.countplot(x='diagnosis', data=cancer_data,palette = "Set2")
        for p in graph.patches:
            graph.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        ax.set_xticklabels(['Benign','Malignant'],fontsize=16)
        ax.set_xlabel('Diagnosis', fontsize=16)
        ax.set_ylabel('Count', fontsize=16)
        st.pyplot(fig2)

    if graphs=='Histogram for Mean Values':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        data_mean.hist(bins=10, figsize=(15, 15))
        plt.show()
        st.pyplot()
        
    if graphs=='Histogram for Standard Error Values':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        data_se.hist(bins=10, figsize=(15, 15))
        plt.show()
        st.pyplot()
    if graphs=='Histogram for Worst Values':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig=data_worst.hist(bins=10, figsize=(15, 15))
        plt.show()
        st.pyplot()
        

        
if menu=='Models Evaluation':
    st.title('Model Evaluation')
    st.write()
    Classifier=st.selectbox("Select Classifier", ( "Logistic Regression", "Random Forest", 'Decision Tree', "Support Vector Machine (SVM)"))

    if Classifier == 'Logistic Regression':
        lr=LogisticRegression(random_state=0)
        lr.fit(X_train,Y_train)
        y_pred=lr.predict(X_test)
        pred_prob = lr.predict_proba(X_test)
        ac_lr=accuracy_score(Y_test,y_pred).round(2)      
        st.write("Logistic Regression :")
        st.write("Accuracy Score:",ac_lr)
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        metrics=st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        plt_metrics(metrics)
        

    if Classifier == 'Random Forest':
        rf=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        rf.fit(X_train,Y_train)
        y_pred=rf.predict(X_test)
        pred_prob = rf.predict_proba(X_test)
        ac_rf=accuracy_score(Y_test,y_pred).round(2)
        st.write("Random Forest :")
        st.write("Accuracy Score:",ac_rf)
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        metrics=st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        plt_metrics(metrics)

                    
    if Classifier == 'Decision Tree':
        dt=DecisionTreeClassifier(random_state=0,criterion="entropy")
        dt.fit(X_train,Y_train)
        y_pred=dt.predict(X_test)
        pred_prob = dt.predict_proba(X_test)
        ac_dt=accuracy_score(Y_test,y_pred).round(2)
        st.write("Decision Tree :")
        st.write("Accuracy Score:",ac_dt)
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        metrics=st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        plt_metrics(metrics)

        

    if Classifier == 'Support Vector Machine (SVM)':
        svm=SVC(kernel='linear',probability=True)
        svm.fit(X_train,Y_train)
        y_pred=svm.predict(X_test)
        pred_prob = svm.predict_proba(X_test)
        ac_svm=accuracy_score(Y_test,y_pred).round(2)       
        st.write("Support Vector Machine (SVM) :")
        st.write("Accuracy Score:",ac_svm)
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        metrics=st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        plt_metrics(metrics)

        

#prediction function
def cancer_classification(data):

    input_arr = np.asarray(data)
    res = input_arr.astype(float)
    input_arr_reshaped = res.reshape(1,-1)
    op = rf_model.predict(input_arr_reshaped)
    if (op[0] == 0):
        return 'Breast Cancer is Benign.'
    else:
        return'Breast Cancer is Malignant.'

def classify():

    st.title('Breast Cancer Prediction')
    
    Radius_mean = st.text_input('Radius Mean')
    Area_mean = st.text_input('Area Mean')
    Compactness_mean = st.text_input('Compactness Mean ')
    Texture_mean = st.text_input('Texture Mean ')
    Smoothness_mean = st.text_input('Smoothness Mean ')
    Symmetry_mean = st.text_input('Symmetry Mean ')
    Area_se = st.text_input('Area SE')
    Fractal_dimension_se = st.text_input('Fractal Dimension SE')
    Perimeter_worst = st.text_input('Perimeter Worst')
    Area_worst = st.text_input('Area Worst')
    Compactness_worst = st.text_input('Compactness Worst')
    Concavity_worst = st.text_input('Concavity Worst')
    Concave_points_worst = st.text_input('Concave Points Worst')
    Texture_worst = st.text_input('Texture Worst')
    Smoothness_worst = st.text_input('Smoothness Worst')
    
    data=[[Radius_mean,Area_mean,Compactness_mean,Texture_mean,Smoothness_mean,Symmetry_mean,Area_se,Fractal_dimension_se,Perimeter_worst,Area_worst,Compactness_worst,Concavity_worst,Concave_points_worst,Texture_worst,Smoothness_worst]]

    diagnosis = ''
    #button for prediction
    if st.button('Breast Cancer Result'):
        diagnosis = cancer_classification(data)
        
    st.success(diagnosis)

#Cancer_Classification    
if menu=='Breast Cancer Prediction':
    rf_model = pickle.load(open('C:/Users/janha/OneDrive/Desktop/ML/breastcancermodel.pkl','rb'))
    classify()


    
