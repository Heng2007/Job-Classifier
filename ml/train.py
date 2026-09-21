"""Train and compare the 5-way posting classifier.

Responsible for: turning labeled postings into TF-IDF features, fitting a
logistic regression, an XGBoost model on them and pytorch model on them, comparing the three, and
saving the winner for the rest of the tool to use. One 5-way classifier.

Inputs: labeled postings from the database (description text and category);
the category list from config; a train/test split.

Outputs: a fitted vectorizer and model saved as .joblib in models/; a macro-F1
score and confusion matrix for each candidate; a row in the model_runs table
recording the run date, model name and macro-F1.

How I'll know it works: both models train without error, macro-F1 is reported
per model, the confusion matrix shows the errors are spread across categories
rather than everything collapsing into one class, and the saved artifact
reloads and predicts on a fresh posting.
"""


import sys 
from pathlib import Path
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import joblib
import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


sys.path.append(str(Path(__file__).resolve().parent.parent))
import config




conn = sqlite3.connect(config.DB_PATH)
df = pd.read_sql_query("SELECT p.title, p.description, l.category "
                        "FROM postings p "
                        "JOIN labels l ON l.posting_id = p.id", conn)

X = df["title"] + " "+ df["description"]
Y = df["category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify= Y, test_size= 0.2, random_state= config.RANDOM_SEED)

vectorizer = TfidfVectorizer()
fitted_x_train = vectorizer.fit_transform(X_train)
fitted_x_test = vectorizer.transform(X_test)

joblib.dump(vectorizer, config.MODELS_DIR +'/vectorizer.joblib')


# Logistic Regression--------------------------------------------------------------------------------------------------
LogReg = LogisticRegression(C = 100, class_weight = "balanced", solver= "lbfgs", max_iter= 1000).fit(fitted_x_train, Y_train)


def logistic_predict(input):
        """Return the predicted outcome(s) as a numpy array of label strings."""
        model_prediction = LogReg.predict(input)
        return model_prediction

# MLP-------------------------------------------------------------------------------------------------------------------


x_train_tensor = torch.tensor(fitted_x_train.todense(), dtype =torch.float32) 
x_test_tensor = torch.tensor(fitted_x_test.todense(), dtype = torch.float32) 

label_encoder = LabelEncoder()

#encode the labels
encoded_y_train = label_encoder.fit_transform(Y_train)
encoded_y_test = label_encoder.transform(Y_test)

#change the datatype of the encoded labels to torch.long
y_train_tensor = torch.tensor(encoded_y_train, dtype = torch.long)
y_test_tensor = torch.tensor(encoded_y_test, dtype = torch.long)

#make it ready for dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle = False )


torch.manual_seed(config.RANDOM_SEED)

#The basic multilayer perception model
#Sequential replaced the class and forward()
mlp_model = nn.modules.Sequential(nn.Linear(len(x_train_tensor[1]), 256),
                                  nn.ReLU(),
                                  nn.Linear(256,64),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.5),
                                  nn.Linear(64,len(label_encoder.classes_)),
                                  )



current_loss = 0
all_losses = []
mlp_model.train()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr = config.LEARNING_RATE)
# optimizer = torch.optim.SGD(mlp_model.parameters(), lr = config.LEARNING_RATE)
counts = np.bincount(y_train_tensor)   

# w_c = N / (K * n_c) where N is total examples, K is number of classes, n_c is the count of class c
# weights = torch.tensor(len(x_train_tensor[1])/(len(label_encoder.classes_) * counts), dtype=torch.float) 
# critation = nn.CrossEntropyLoss(weight= weights)
critation = nn.CrossEntropyLoss()
        
def train(model: nn.Sequential, n_epoch = 10):
        current_loss = 0
        all_losses = []


        for epoch in range(1, n_epoch+1):
                model.zero_grad()
                for index, (x_batch, y_batch) in enumerate(train_loader):
                        output = model.forward(x_batch)
                        loss = critation(output, y_batch)

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        current_loss += loss.item()
                average_loss = current_loss/len(train_loader)
                # print(average_loss)
                all_losses.append(average_loss)
                current_loss = 0

train(mlp_model)
mlp_model.train(False)


joblib.dump(mlp_model,config.MODELS_DIR+'/mlp.joblib')
joblib.dump(label_encoder, config.MODELS_DIR + '/encoder.joblib')


with torch.no_grad():
    predictions = mlp_model(x_test_tensor).argmax(1)


con = sqlite3.connect(config.DB_PATH)
cur = con.cursor()

mlp_result = classification_report(y_test_tensor, predictions, target_names=label_encoder.classes_, output_dict = True)

cur.execute('INSERT INTO model_runs(run_date, model_name, macro_f1, notes) VALUES(?,?,?,?)',
            (datetime.date.today().isoformat(), config.MODEL_NAME_MLP, mlp_result['macro avg']['f1-score'], " First version Multilayer Perceptron" )) # type: ignore

LogReg_result = classification_report(Y_test, LogReg.predict(fitted_x_test), output_dict= True)

print("mlp save")

cur.execute('INSERT INTO model_runs(run_date, model_name, macro_f1, notes) VALUES(?,?,?,?)',
            (datetime.date.today().isoformat(), config.MODEL_NAME_LOG, LogReg_result['macro avg']['f1-score'], " First version  Logistic Regression" )) # type: ignore
print("logreg saved")

con.commit()