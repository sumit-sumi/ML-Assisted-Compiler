# ML-Assisted-Compiler
Ml  assisted Compiler
 To help you understand every code in each file, what libraries are used, and what 
each block of code does, I will break down the entire process and explain the 
purpose of each file and the individual components of the code. Here's an in
depth explanation for every file involved in your MLAssisted Compiler project.
 File 1: 
main.py (Compiler & ML Logic)
 Libraries Used:
  re Regular Expressions):
 oblib :
 Used for tokenizing the input code by matching patterns (like keywords, 
variables, and operators).
  j
 Used to load the pre-trained machine learning model from a 
predictions.
  p
 andas :
 .pkl file for 
Used for handling data structures (like DataFrames) for storing features 
extracted from the input code.
 Code Breakdown:
 python
 CopyEdit
 import re
 import joblib
 import pandas as pd
 Purpose Importing necessary libraries for tokenization (
 ( j
 oblib ), and data manipulation (
 re ), machine learning 
pandas ).
 1
 Ml assisted Compiler
python
 CopyEdit
 def tokenize_code(code):
 tokens = re.findall(r'\w+|^\w\s]', code)
 return tokens
 Purpose:
 This function takes the input code as a string and tokenizes it into 
individual tokens (such as keywords, variables, operators, etc.).
 re.findall() is used to find all matches for the regex pattern 
\w+|^\w\s] , which 
matches any word or non-whitespace character (e.g., keywords, 
operators, punctuation).
 python
 CopyEdit
 def extract_features(tokens):
 features = {
 "length": len(tokens),
 "semicolons": tokens.count(';'),
 "parentheses": tokens.count('(') + tokens.count(')')
 }
 return features
 Purpose:
 This function extracts a few basic features from the tokenized code that 
will be used for the machine learning model.
 It counts:
 The length of the token list (total number of tokens).
 The number of semicolons ( ; ).
 2
 Ml assisted Compiler
The number of parentheses (( and )).
 python
 CopyEdit
 def check_code_issues(code):
 tokens = tokenize_code(code)
 features = extract_features(tokens)
 # Load the pre-trained machine learning model
 model = joblib.load('model.pkl')
 es
 # Predict if the code is likely correct or erroneous based on extracted featur
 prediction = model.predict(pd.DataFrame([features]))
 if prediction  1
 return "Code is likely correct!"
 else:
 return "Code is likely erroneous!"
 Purpose:
 This function combines tokenization, feature extraction, and machine 
learning prediction to analyze the input code.
 It uses the 
model.pkl file (a pre-trained machine learning model) to predict 
whether the code is correct or erroneous based on the extracted features 
(like length, semicolon count, and parentheses count).
 If the prediction is 1 , it means the code is likely correct; otherwise, itʼs 
erroneous.
 File 2: 
app.py (Streamlit UI)
 Libraries Used:
 3
 Ml assisted Compiler
 s
 treamlit :
 Used to create a user-friendly web interface where users can enter code 
and view predictions.
 Code Breakdown:
 python
 CopyEdit
 import streamlit as st
 from compiler.lexer import tokenize
 Purpose:
 streamlit  Used for building the web interface where users interact with the 
application.
 tokenize  Imports the tokenization function from the lexer module for code 
analysis (part of the compiler logic).
 python
 CopyEdit
 def main():
 st.title("MLAssisted Compiler")
 code_input = st.text_area("Enter your code")
 if st.button("Check Code"):
 result = check_code_issues(code_input)
 st.write(result)
 Purpose:
 The main function defines the structure of the Streamlit app.
 st.title()  Sets the title of the web page.
 st.text_area()  Creates a text box where users can enter their code.
 4
 Ml assisted Compiler
st.button()  Adds a button for the user to submit the code for analysis.
 check_code_issues(code_input)  Calls the function from 
main.py to analyze the 
code.
 st.write()  Displays the result of the analysis (whether the code is correct or 
erroneous).
 File 3: 
lexer.py (Compiler Logic - Tokenization)
 Libraries Used:
  re Regular Expressions):
 Used for writing regex patterns to match different tokens in the code.
 Code Breakdown:
 python
 CopyEdit
 import re
 Purpose Import the 
re library to handle regular expressions, which are 
essential for breaking down the code into meaningful components (tokens).
 python
 CopyEdit
 def tokenize(code):
 tokens = re.findall(r'\w+|^\w\s]', code)
 return tokens
 Purpose:
 The 
tokenize() function breaks down the input code into individual tokens.
 The regex pattern 
\w+|^\w\s] matches all words (like keywords, variables) 
and non-whitespace characters (like operators and punctuation).
 5
 Ml assisted Compiler
The result is a list of tokens that can be analyzed further by the machine 
learning model.
 File 4: 
model.py (ML Model - Training)
 Libraries Used:
  p
 andas :
 klearn :
 Used to load and manipulate the dataset (a CSV file containing examples 
of correct and incorrect code features).
  s
 DecisionTreeClassifier  A machine learning model that classifies code as correct 
or erroneous based on extracted features.
 train_test_split  Splits the dataset into training and testing sets for evaluation.
 Code Breakdown:
 python
 CopyEdit
 import pandas as pd
 from sklearn.tree import DecisionTreeClassifier
 from sklearn.model_selection import train_test_split
 import joblib
 Purpose:
 pandas  For loading and processing the dataset.
 DecisionTreeClassifier  A classifier to predict whether the code is correct or 
erroneous based on features.
 train_test_split  Splits the data for training and testing.
 joblib  Saves the trained model to a 
.pkl file.
 6
 Ml assisted Compiler
python
 CopyEdit
 # Load dataset and split into features and target
 data = pd.read_csv('dataset.csv')
 X  data.drop('label', axis=1  # Features
 y = data['label']  # Target (correct or erroneous)
 # Split the data into training and testing sets
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2
 # Train the model
 clf  DecisionTreeClassifier()
 clf.fit(X_train, y_train)
 # Save the trained model
 joblib.dump(clf, 'model.pkl')
 Purpose:
 Loading the dataset A CSV file (
 dataset.csv ) that contains feature data and 
corresponding labels (correct or erroneous).
 Splitting the data Divides the data into training and testing sets 80% 
training, 20% testing).
 Training the model A 
DecisionTreeClassifier is used to train the model.
 Saving the model The trained model is saved as 
model.pkl for later use.
 File 5: 
dataset.csv (Dataset for Training the Model)
 This CSV file contains sample data for training the machine learning model. Each 
row corresponds to a piece of code with extracted features (e.g., number of 
tokens, semicolons, parentheses) and a label indicating whether the code is 
correct or erroneous.
 Sample Structure:
 7
 Ml assisted Compiler
length
 10
 8
 12
 semicolons
 2
 1
 3
 parentheses
 1
 2
 3
 length  Number of tokens in the code.
 semicolons  Number of semicolons ( ; ).
 label
 1
 0
 1
 parentheses  Total number of parentheses (( and )).
 label  Target variable: 1 (correct) or 
0 (erroneous).
 Summary of the Project Flow:
  Lexer (lexer.py) Tokenizes the input code using regex.
  Feature Extraction (main.py) Extracts simple features (length, semicolons, 
parentheses) from the tokenized code.
  Machine Learning (model.py) Trains a decision tree classifier on labeled data 
and predicts if the code is correct or erroneous.
  UI (app.py) Provides an interface for users to input code and see the results 
of the analysis.
 If you need further explanations or clarifications for any specific part of the code, 
feel free to ask!
 8
 Ml assisted Compiler
