from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import pickle

# Read the dataset
df = pd.read_csv("/Users/micah.mathews/Documents/wager_ai/lol_data/data_cleaning/cleaned_data.csv")

# Select the features and target variable
X = df[['difference_avg_player_champ_winrate', 'difference_avg_champion_patch_winrate', 'difference_avg_counter_winrate']]
y = df['result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the results for the test set
y_pred = model.predict(X_test)

# Save the model to a file
with open("/Users/micah.mathews/Documents/wager_ai/model/lol_predictions_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

classification_rep = classification_report(y_test, y_pred)
print(f'Classification Report: \n{classification_rep}')

confusion_mat = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{confusion_mat}')
