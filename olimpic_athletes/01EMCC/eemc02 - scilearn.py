import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump

"""
EEMC - Estimating Medal Count for a Country

model trainig: linear regression
"""

df_medals = pd.read_csv('medals_by_country.csv')

# encode 'Season': Summer = 0; Winter = 1
df_medals['Season'] = df_medals['Season'].map({'Summer': 0, 'Winter': 1})

# One-hot encode 'Team' using Scikit-Learn
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
team_encoded = encoder.fit_transform(df_medals[['Team']])

dump(encoder, 'team_encoder.joblib')

team_encoded_df = pd.DataFrame(team_encoded, columns=encoder.get_feature_names_out(['Team']))

df_medals = pd.concat([df_medals.drop('Team', axis=1), team_encoded_df], axis=1)

X = df_medals.drop('Medal Count', axis=1)
y = df_medals['Medal Count']

# Split and SHUFFLE the data into training and testing sets
# test_size=0.2 - 20% for testing
# random_state - reproducibility of splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

residuals = y_test - y_pred_test

# residuals
plt.scatter(y_pred_test, residuals)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Medal Count')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='dashed')
plt.show()

# predictions vs actual values
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.title('Actual vs Predicted Medal Counts')
plt.xlabel('Actual Medal Count')
plt.ylabel('Predicted Medal Count')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()


plt.hist(residuals, bins=30)
plt.title('Histogram of Residual')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

dump(model, 'olimpic_medal_predictor.joblib')
































