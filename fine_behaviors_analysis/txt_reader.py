import re
import os
import ast

def read_txt(file):
    with open(file, 'r') as f:
        data = f.readlines()
    out = {}
    for line in data:
        if line.strip().startswith('#'):
            continue
        line = line.strip()
        k, v = line.split(':')
        if v.startswith('['):
            v = ast.literal_eval(v)
        elif v.isdigit() or v.replace('-', '1').isdigit(): # Catches integers (including negative ones)
            v = int(v)
        elif v.replace('.', '1').replace('-', '1').replace('e', '1').isdigit():
            v = float(v)
        out[k] = v
    return out

if __name__ == "__main__":
    out = read_txt('fine_behaviors_analysis/models/more_test_params.txt')
    print(out)