import pandas as pd
import numpy as np 
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# parameters

C=0.01
output_file=f"model_C={C}.bin"

# data prepration

data="customer.csv"
df=pd.read_csv(data)

df.columns=df.columns.str.lower().str.replace(" ","_")

categorical_columns=list(df.dtypes[df.dtypes=="object"].index)

for c in categorical_columns: 
    df[c]=df[c] .str.lower().str.replace(" ","_")

df.totalcharges=pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges=df.totalcharges.fillna(0)
df.churn=(df.churn =="yes").astype(int)    

df_full_train,df_test=train_test_split(df,test_size=0.2,random_state=1)

numerical =['tenure','monthlycharges', 'totalcharges']
categorical =['gender', 'seniorcitizen', 'partner', 'dependents','phoneservice', 
            'multiplelines', 'internetservice','onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport','streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
            'paymentmethod']

# training functions
def training(df_train,y_train, C):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv=DictVectorizer(sparse=False)
    X_train=dv.fit_transform(dicts)

    model=LogisticRegression(C=C,max_iter=10000)
    model.fit(X_train,y_train)

    return dv,model

def predict(df,dv,model):
    dicts = df[categorical + numerical].to_dict(orient="records")
    X=dv.transform(dicts)

    y_pred=model.predict_proba(X)[:,1]

    return y_pred


# final train

dv, model = training(df_full_train, df_full_train.churn.values, C)
y_pred = predict(df_test, dv, model)


# save the model

with open(output_file,"wb") as f_out:
    pickle.dump((dv,model),f_out)

