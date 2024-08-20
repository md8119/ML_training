from joblib import load
import pandas as pd

# Load the model and the encoder
model = load('olimpic_medal_predictor.joblib')
encoder = load('team_encoder.joblib')

# Prepare new data
new_data = pd.DataFrame({
    'Team': ['Unated States-1'],
    'Year': [2024],
    'Season': ['Summer']
})

# Properly encode 'Season'
new_data['Season'] = new_data['Season'].map({'Summer': 0, 'Winter': 1})

# Encode 'Team' using the loaded encoder
team_encoded = encoder.transform(new_data[['Team']])
team_encoded_df = pd.DataFrame(team_encoded, columns=encoder.get_feature_names_out())

# Concatenate with other data
new_data_prepared = pd.concat([new_data.drop('Team', axis=1), team_encoded_df], axis=1)

# Ensure all expected columns are present (as the model was trained with these)
expected_cols = model.feature_names_in_  # This assumes you're using a version of sklearn that supports feature_names_in_
missing_cols = set(expected_cols) - set(new_data_prepared.columns)
for col in missing_cols:
    new_data_prepared[col] = 0

new_data_prepared = new_data_prepared[expected_cols]  # Ensure the columns order

# Predict
predicted_medals = model.predict(new_data_prepared)
print(f'Predicted medals: {predicted_medals[0]}')
