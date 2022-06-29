from matplotlib.pyplot import step
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import xgboost as xgb
from sklearn.model_selection import train_test_split


st.title("Economic Freedom Index Predition using XGBoost")
st.markdown("Out of all the models that we tested, **XGBoost** was giving us the best result. Hence we decided to make a predictor function which based on user inputs values of our Input Features")


df=pd.read_csv('XGBOOST_data2.csv')

see_data = st.expander('You can click here to see the raw data first 👉')
with see_data:
    st.dataframe(data=df.reset_index(drop=True))


X=df.drop(['Old Score Category', 'Score Category'], axis=1)
y=df['Score Category']

# st.dataframe(X)
# st.dataframe(y)

# st.write('X',X.shape)
# st.write('y',y.shape)

# st.write(y.unique())


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=100)

# df.shape

# X.shape
# y.shape

# X_train.shape
# X_test.shape
# y_train.shape
# y_test.shape

param = {}
param['booster'] = 'gbtree'
param['n_estimators'] = 50
param['objective'] = 'binary:logistic'
param["eval_metric"] = "error"
param['eta'] = 0.3
param['gamma'] = 0
param['max_depth'] = 3
param['min_child_weight']=1
param['max_delta_step'] = 0
param['subsample']= 1
param['colsample_bytree']=1
param['silent'] = 1
param['seed'] = 0
param['base_score'] = 0.5

model = xgb.XGBClassifier(param)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

p1 = st.number_input('Property Rights Value', value=30.0, step=0.01, key=1)
p2 = st.number_input('Fiscal Freedom Value', value=92.5, step=0.01, key=2)
p3 = st.number_input('Government Spending Value', value=75.1, step=0.01, key=3)
p4 = st.number_input('Corporate Tax Rate (%)', value=10.0, step=0.01, key=4)
p5 = st.number_input('Tax Burden % of GDP', value=23.3, step=0.01, key=5)
p6 = st.number_input('Population (Millions)', value=3.2, step=0.01, key=6)
p7 = st.number_input('GDP Growth Rate (%)', value=2.0, step=0.01, key=7)
p8 = st.number_input('Unemployment (%)', value=13.5, step=0.01, key=8)
p9 = st.number_input('FDI Inflow (Millions)', value=-0.4, step=0.01, key=9)
p10 = st.number_input('Public Debt (% of GDP)', value=58.9, step=0.01, key=10)
p11 = st.number_input('Asia-Pacific', value=0.0, step=0.01, key=11)
p12 = st.number_input('Europe', value=1.0, step=0.01, key=12)
p13 = st.number_input('Middle East / North Africa', value=0.0, step=0.01, key=13)
p14 = st.number_input('North America', value=0.0, step=0.01, key=14)
p15 = st.number_input('South and Central America / Caribbean', value=0.0, step=0.01, key=15)
p16 = st.number_input('Sub-Saharan Africa', value=0.0, step=0.01, key=16)

X = [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16]]
X=pd.DataFrame(X)
X.columns=['Property Rights', 'Fiscal Freedom ', 'Gov Spending',
       'Corporate Tax Rate (%)', 'Tax Burden % of GDP',
       'Population (Millions)', 'GDP Growth Rate (%)', 'Unemployment (%)',
       'FDI Inflow (Millions)', 'Public Debt (% of GDP)', 'Asia-Pacific',
       'Europe', 'Middle East / North Africa', 'North America',
       'South and Central America / Caribbean', 'Sub-Saharan Africa']

y_pred = model.predict(X)

st.write('With the following input values, the county will fall in the category:',y_pred[0])