import pandas as pd
from sklearn import preprocessing   

df = pd.read_csv("set_b.csv")
df = df.drop("sublabel", axis=1)
df = df.replace(["extrastole", "murmur"], "abnormal")
for index, row in df.iterrows():
    row['fname'] = row['fname'].replace('set_b/Btraining_','')
    row['fname'] = row['fname'].replace('extrastole_','extrastole__')
    row['fname'] = row['fname'].replace('murmur_','murmur__')
    row['fname'] = row['fname'].replace('normal_','normal__')
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])

df.to_csv("pre-set_b.csv")
