import pandas as pd
files = ["locations", "incidents", "public_complaints"]
for f in files:
    d = pd.read_csv(f"data/{f}.csv")
    print("=" * 50, f, d.shape)
    print(d.dtypes)
    print("missing:\n", d.isna().sum())
    print(d.head(5).to_string())
    for col in d.select_dtypes("object").columns:
        if d[col].nunique() < 15:
            print(col, d[col].value_counts().to_dict())
