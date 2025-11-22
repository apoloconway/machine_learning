import pickle

# parameter
input_file='05-deployment/model_C=0.01.bin'

# load model

with open(input_file,"rb") as f_in:
    dv,model=pickle.load(f_in)


# example
customer={'gender': 'female',
 'seniorcitizen': 0,
 'partner': 'yes',
 'dependents': 'no',
 'phoneservice': 'no',
 'multiplelines': 'no_phone_service',
 'internetservice': 'dls',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'yes',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'tenure': 4,
 'monthlycharges': 29.85,
 'totalcharges': 29.85}

X_0=dv.transform([customer])
print("input: ",customer)
print("churn probability: ",model.predict_proba(X_0)[0,1])


