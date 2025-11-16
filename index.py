import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('./data.csv')

df['tone_vector'] = df['tone_vector'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['tone_vector'])
print(f"Labels: {mlb.classes_}")