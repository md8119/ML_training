import pandas as pd
import matplotlib.pyplot as plt


# load the CSV file into the DataFrame
df = pd.read_csv('athlete_events.csv')

# display the first few rows
print('### HEAD ###')
print(df.head())

# check for missing values
# NA or NaN are considered as missed
print('\n### MISSING VALUES ###')
print(df.isnull().sum())

# summary of the data
print('\n### DESCRIBE ###')
print(df.describe())
print('\n### INFO ###')
print(df.info())

# create a histogram for the 'Age' column
df['Age'].hist(bins=40, edgecolor='white')
plt.title('Distribution of Athlete Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# create 'Athlets by Country' plot
athlets_by_country = df['Team'].value_counts().head(15)
ax = athlets_by_country.plot(kind='bar', figsize=(10,6), color='skyblue')
plt.title('Number of Athlets by Country (Top 10)')
plt.xlabel('Country')
plt.ylabel('Number of Athlets')
plt.xticks(rotation=45)
# add the numbers above the bars
for index, value in enumerate(athlets_by_country):
    plt.text(index, value, str(value), ha='center', va='bottom')
plt.show()