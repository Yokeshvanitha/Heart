# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Loading the dataset
heart = pd.read_csv("hearts.csv")  # Ensure the file path is correct
print(heart)

# Data Preprocessing
# Encoding categorical features
label_encoder = LabelEncoder()

# Specify the columns that are categorical
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_columns:
    heart[col] = label_encoder.fit_transform(heart[col])
    print(heart)

# Checking the data types to confirm all are numeric
print("Data types after encoding:\n", heart.dtypes)

# Check for any missing values
print("Missing values:\n", heart.isnull().sum())

# Separate features (X) and target variable (y)
X = heart.drop(columns=['HeartDisease'])  # Replace 'HeartDisease' with your target column name
y = heart['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)#split the values

#X_train  - 80% Input Data
#Y_train  - 80% Output Data
#X_test  - 20% Input Data
#X_test  - 20% Output Data

print("DF",heart.shape)
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("Y_train",y_train.shape)
print("Y_test",y_test.shape)


# Model Training
model = GaussianNB()
model.fit(X_train, y_train)

#model evaluation
y_pred=model.predict( X_test)

print("y_pred",y_pred)
print("Y_test",y_test)

from sklearn.metrics import accuracy_score
print('ACCURACY IS',accuracy_score(y_test,y_pred))

#model Prediction

import pandas as pd

test_data = pd.DataFrame([[29,0,2,100,106,1,2,80,1,1,1]],columns=X_train.columns)

#Make the Prediction
testPrediction = model.predict(test_data)

#check the prediction result
if testPrediction[0]==1:
    print("THE PATIENT HAVE HEART DISEASE,PLEASE CONSULT THE DOCTOR")
else:
    print("THE PATIENT IS NORMAL")


# Save the model to a .pkl file
  
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved to trained_model.pkl")
