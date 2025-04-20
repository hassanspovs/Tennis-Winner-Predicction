import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\hassa\OneDrive\Documents\Visual Studio 2022\ML PROJECTS\atp_tennis.csv")

df = df.dropna()

label_encoder = LabelEncoder()
df['Player_1_encoded'] = label_encoder.fit_transform(df['Player_1'])
df['Player_2_encoded'] = label_encoder.fit_transform(df['Player_2'])
df['Winner_encoded'] = label_encoder.fit_transform(df['Winner'])

if 'Surface' in df.columns:
    df['Surface'] = df['Surface'].map({'Hard': 1, 'Clay': 2, 'Grass': 3})
else:
    df['Surface'] = 0  # if missing, fill with 0

df['rank_diff'] = df['Rank_1'] - df['Rank_2']
df['pts_diff'] = df['Pts_1'] - df['Pts_2']
df['odd_diff'] = df['Odd_1'] - df['Odd_2']

features = ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Surface',
            'Player_1_encoded', 'Player_2_encoded', 'rank_diff', 'pts_diff', 'odd_diff']
X = df[features]
y = df['Winner_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

example_match = [[25, 30, 1800, 1600, 1.9, 2.1, 1, 150, 200, -5, 200, -0.2]]
predicted_winner_encoded = model.predict(example_match)[0]
predicted_winner = label_encoder.inverse_transform([predicted_winner_encoded])[0]
print(f"🏆 Predicted Winner: {predicted_winner}")
