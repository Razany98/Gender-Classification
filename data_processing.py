#importing the libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf 

#deriving the dataset 
df = pd.read_csv('/content/drive/MyDrive/voice.csv')

#Reading the dataset & Pre-Processing

df.isnull().sum()
df.isna().sum()

print("gender set dimensions : {}".format(df.shape)) #Dimension of the dataset

#Visualization

df['label'].value_counts().plot.bar()

#1 for Male and 0 for Female
df.label = [1 if each == "male" else 0 for each in df.label]
#1 for Male and 0 for Female

#Splitting & Scaling the dataset

y = df['label'].copy()
X = df.drop('label', axis=1).copy() #Drop irrelevant feature

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42) #split to training and testing dataset 

X.shape #After dropping the irrelevant feature we notice that we have 3168 rows and 20 columns 

#Modeling, used 2 hidden layers 
inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

model.evaluate(X_test, y_test)


