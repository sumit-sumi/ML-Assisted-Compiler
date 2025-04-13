import re
import joblib
import pandas as pd

# Load trained model
model = joblib.load('ml_model/model.pkl')

# Tokenizer function
def tokenize_code(code):
    # Simple regex tokenizer for demo purposes
    tokens = re.findall(r'\w+|[^\s\w]', code)
    return tokens

# Feature extractor
def extract_features(code):
    tokens = tokenize_code(code)
    length = len(code)
    semicolons = tokens.count(';')
    parentheses = tokens.count('(') + tokens.count(')')
    return pd.DataFrame([[length, semicolons, parentheses]], columns=['length', 'semicolons', 'parentheses'])

# Function to check for missing elements (like keywords before variable declaration)
def check_code_issues(code):
    issues = []
    
    # Check if variable is declared without a keyword (e.g., int, float, etc.)
    # We'll look for patterns like `variable = value` but without a keyword in front.
    pattern = r'([a-zA-Z_]\w*)\s*=\s*[^;]+'
    matches = re.findall(pattern, code)

    for match in matches:
        # Check if the match is a simple variable assignment without a keyword like `int`
        if not any(keyword in code for keyword in ['int', 'float', 'char', 'double', 'string', 'boolean']):
            issues.append(f"Keyword is missing before variable '{match}'.")

    # Check if semicolon is missing
    if not code.strip().endswith(';'):
        issues.append("Semicolon ';' is missing at the end.")
    
    # Add more checks here based on other possible patterns (e.g., missing parentheses, brackets, etc.)
    return issues

# Main
if __name__ == '__main__':
    code_input = input("Enter your code: ")
    tokens = tokenize_code(code_input)
    print(f"Tokens: {tokens}")

    # Check for specific code issues
    issues = check_code_issues(code_input)
    
    if issues:
        for issue in issues:
            print(f"Code Issue: {issue}")
    else:
        features = extract_features(code_input)
        prediction = model.predict(features)

        if prediction[0] == 1:
            print("ML Check: Looks Correct ✅")
        else:
            print("ML Check: Likely Error ❌")
