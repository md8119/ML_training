import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""
EEMC - Estimating Medal Count for a Country

model trainig: linear regression
"""

df_medals = pd.read_csv('medals_by_country.csv')

# encode 'Season': Summer = 0; Winter = 1
df_medals['Season'] = df_medals['Season'].map({'Summer': 0, 'Winter': 1})

# concatenate relevant columns before shuffling
data = df_medals[['Team', 'Year', 'Season', 'Medal Count']].values

# shuffle the data
np.random.shuffle(data)

# split the data into the training and testing sets
# 80% training, 20% test
splitter = int(len(data) * 0.8)
data_train = data[:splitter]
data_test = data[splitter:]

# separate features and target
teams_train = data_train[:, 0]
teams_test = data_test[:, 0]

X_train = data_train[:, 1:-1]
X_test = data_test[:, 1:-1]

y_train = data_train[:, -1]
y_test = data_test[:, -1]

# one-hot encode the 'Team' column
teams_train_encoded = pd.get_dummies(teams_train, dtype=int)
teams_test_encoded = pd.get_dummies(teams_test, dtype=int)

teams_test_encoded = teams_test_encoded.reindex(columns=teams_train_encoded.columns,\
                                                fill_value=0)
X_train = np.concatenate([teams_train_encoded, X_train], axis=1)
X_test = np.concatenate([teams_test_encoded, X_test], axis=1)

# Because both arrays were of the 'object' type here is conversion for float64
X_train = np.array(pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce')).astype(np.float64)
y_train = pd.to_numeric(y_train, errors='coerce').astype(np.float64)


### TRAINING ###
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialise weights (w) to zeros
w = np.zeros(X_train.shape[1], dtype=np.float64)
# initialise the bias (b) to zero
b = 0

# set hyperparameters
learning_rate = 0.001
n_iterations = 1000


print(f'learning_rate = {learning_rate}')

costs = []
# gradient descent loop
for i in range(n_iterations):
    # 1 calculate predictions
    y_pred = np.dot(X_train, w) + b
    
    # 2 compute the cost
    error = y_pred - y_train
    cost = (1 / (2 * len(y_train))) * np.dot(error.T, error)
    
    # 3 calculate gradients
    dw = (1 / len(y_train)) * np.dot(X_train.T, error)
    db = (1 / len(y_train)) * np.sum(error)
    
    # 4 update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # save the cost value for plotting
    costs.append(cost)
    
    if i % 100 == 0:
        print(f'Iteration {i}: Cost {cost}')
        
        
# Visualising data
y_pred_test = np.dot(X_test, w) + b
# Cost over iterations
plt.plot(range(n_iterations), costs)
plt.title('Cost Reduction Over Interations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Predictions vs Actual Values
plt.scatter(y_test, y_pred_test)
plt.title('Actual vs Predicted Medal Counts')
plt.xlabel('Actual Medal Count')
plt.ylabel('Predicted Medal Count')
plt.show()

























