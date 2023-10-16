import pickle

# loads DictVectorizer and model
with open("./dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)
with open("./model1.bin", "rb") as f_in:
    model1 = pickle.load(f_in)

# client (input data) to predict
client = {"job": "retired", "duration": 445, "poutcome": "success"}

# transforms using DictVectorizer
client_enc = dv.transform(client)

# predicts using Logistic Regression model
# saves only the prediction probability for positive class
y_hat = model1.predict_proba(client_enc)[:, 1]

print(y_hat)
