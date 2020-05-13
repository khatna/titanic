import numpy as np
import pandas as pd
import rfc
import csv

rawtr = pd.read_csv('./train_final.csv')
train = rawtr.drop(columns=[
    'Name', 'Pclass', 'Age', 'Survived', 'PassengerId', 'Cabin', 'Ticket', 'Title', 'Embarked'
])

rawte = pd.read_csv('./test_final.csv')
test = rawte.drop(columns=[
    'Name', 'Pclass', 'Age', 'Cabin', 'Ticket', 'Title', 'Embarked'
])

# Training the model
ytr = rawtr['Survived'].to_numpy()

Xtr = train.to_numpy()
clf = rfc.RandomForest(Xtr, ytr, 890, 500)

# Make predictions
Xte = test.to_numpy()

with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['PassengerId','Survived'])
    for i in range(len(Xte)):
        pid  = Xte[i,0]
        x    = Xte[i,1:]
        pred = clf.classify(x)
        writer.writerow([pid, pred])