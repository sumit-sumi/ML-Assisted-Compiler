import joblib

model = joblib.load("ml_model/model.pkl")

def check_errors(code_snippet):
    # Convert code to feature vector (very simple version)
    features = [len(code_snippet), code_snippet.count(';'), code_snippet.count('(')]
    prediction = model.predict([features])
    return "Correct" if prediction[0] == 1 else "Likely Error"
