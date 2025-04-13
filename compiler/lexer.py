import re

def tokenize(code):
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+|==|<=|>=|!=|[+\-*/=;]', code)
    return tokens
