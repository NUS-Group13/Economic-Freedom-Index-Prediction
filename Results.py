import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 


def main():
    st.title("Economic Freedom Index Predition using Different ML Models")
    st.sidebar.title("Selection of Classifier")
    # st.markdown("Are your mushrooms edible or poisonous?🍄")
    # st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")

    def load_data():
        data = pd.read_csv('preprocessed_data.csv')
        return data

    df = load_data()

    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=df.reset_index(drop=True))


    # @st.cache(persist=True)
    # def split(df):
    #     y = df.type
    #     x = df.drop(columns =['type'])
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    #     return x_train, x_test, y_train, y_test


    # def plot_metrics(metrics_list):
    #     st.set_option('deprecation.showPyplotGlobalUse', False)



    #     if 'Confusion Matrix' in metrics_list:
    #         st.subheader("Confusion Matrix") 
    #         plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
    #         st.pyplot()
        
    #     if 'ROC Curve' in metrics_list:
    #         st.subheader("ROC Curve") 
    #         plot_roc_curve(model, x_test, y_test)
    #         st.pyplot()

    #     if 'Precision-Recall Curve' in metrics_list:
    #         st.subheader("Precision-Recall Curve")
    #         plot_precision_recall_curve(model, x_test, y_test)
    #         st.pyplot()

    #st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Select your Classifier", ("Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine (SVM)", "Naive Bayes", "Gradient Boosting", "XGBoost", "ANN"))

    X_options = st.multiselect(
     'Select the Input Features',
     ['Property Rights', 'Fiscal Freedom', 'Gov Spending', 'Corporate Tax Rate (%)', 'Tax Burden percentage of GDP', 'Population (Millions)', 'GDP Growth Rate (%)', 'Unemployment (%)', 'FDI Inflow (Millions)', 'Public Debt (percentage of GDP)', 'Asia-Pacific', 'Europe', 'Middle East / North Africa', 'North America', 'South and Central America / Caribbean', 'Sub-Saharan Africa'],
     ['Property Rights', 'Fiscal Freedom', 'Gov Spending', 'Corporate Tax Rate (%)', 'Tax Burden percentage of GDP', 'Population (Millions)', 'GDP Growth Rate (%)', 'Unemployment (%)', 'FDI Inflow (Millions)', 'Public Debt (percentage of GDP)', 'Asia-Pacific', 'Europe', 'Middle East / North Africa', 'North America', 'South and Central America / Caribbean', 'Sub-Saharan Africa'], key='x_multiselect')

    st.subheader('Your selected values for X :')
    for x in range(len(X_options)):
        st.write( X_options[x])

    Y_options = st.multiselect(
     'Select the Output Features',
     ['Year Score'],
     ['Year Score'], key='y_multiselect')

    st.subheader('Your selected values for Y :')
    for x in range(len(Y_options)):
        st.write( Y_options[x])


    st.subheader('Results for the selected Classifer - ')
    if classifier == 'Logistic Regression':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 81)
        st.write(f'Precision =', 82)
        st.write(f'Recall =', 83)
        st.write(f'F-Score =', 82)

    if classifier == 'Decision Tree':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 84)
        st.write(f'Precision =', 86)
        st.write(f'Recall =', 86)
        st.write(f'F-Score =', 86)
    
    if classifier == 'Random Forest':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 91)
        st.write(f'Precision =', 82)
        st.write(f'Recall =', 83)
        st.write(f'F-Score =', 82)
    
    if classifier == 'Support Vector Machine (SVM)':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 85)
        st.write(f'Precision =', 83)
        st.write(f'Recall =', 83)
        st.write(f'F-Score =', 83)

    if classifier == 'Naive Bayes':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 59)
        st.write(f'Precision =', 58)
        st.write(f'Recall =', 59)
        st.write(f'F-Score =', 57)

    if classifier == 'Gradient Boosting':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 86)
        st.write(f'Precision =', 88)
        st.write(f'Recall =', 87)
        st.write(f'F-Score =', 88)
    
    if classifier == 'XGBoost':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 92)
        st.write(f'Precision =', 93)
        st.write(f'Recall =', 94)
        st.write(f'F-Score =', 94)
    
    if classifier == 'ANN':
        st.write(f'Classifier = {classifier}')
        st.write(f'Accuracy =', 64)
        st.write(f'Precision =', 63)
        st.write(f'Recall =', 65)
        st.write(f'F-Score =', 65)
    # if classifier == 'Support Vector Machine (SVM)':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C')
    #     kernel = st.sidebar.radio("Kernel",("rbf", "linear"), key='kernel')
    #     gamma = st.sidebar.radio("Gamma (Kernel Coefficient", ("scale", "auto"), key = 'gamma')
    #     metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    #     if st.sidebar.button("Classfiy", key='classify'):
    #         st.subheader("Support Vector Machine (SVM Results")
    #         model = SVC(C=C, kernel=kernel, gamma=gamma)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)

    # if classifier == 'Logistic Regression':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    #     max_iter = st.sidebar.slider("Maxiumum number of interations", 100, 500, key='max_iter')
    #     metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    #     if st.sidebar.button("Classfiy", key='classify'):
    #         st.subheader("Logistic Regression Results")
    #         model = LogisticRegression(C=C, max_iter=max_iter)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)



    # if classifier == 'Random Forest':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     n_estimators  = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    #     max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    #     bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True','False'), key='bootstrap')
    #     metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    #     if st.sidebar.button("Classfiy", key='classify'):
    #         st.subheader("")
    #         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)

    df2 = pd.read_csv('model_scores.csv')

    if st.sidebar.checkbox("Show combined data", False):
        st.subheader("Evaluation Scores of all the Models Compared")
        st.dataframe(df2)


if __name__ == '__main__':
    main()

