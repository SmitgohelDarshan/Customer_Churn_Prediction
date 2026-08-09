import json

nb_path = r"A:\B.TECH\ML PROJECT\EDA.ipynb"
keywords = ['LabelEncoder', 'get_dummies', 'StandardScaler', 'LogisticRegression',
            'pickle', 'pkl', 'label', 'drop_first', 'scaler', 'fit_transform',
            'train_test_split', 'import', 'read_csv']

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(kw in src for kw in keywords):
        ctype = cell['cell_type']
        print(f"=== Cell {i} ({ctype}) ===")
        print(src[:4000])
        print()
