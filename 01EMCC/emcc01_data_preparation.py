import pandas as pd

"""
EEMC - Estimating Medal Count for a Country

data preparation
"""

df = pd.read_csv('eemc.csv')

# Group by 'Team', 'Year', and 'Season' and count the medals
medals_by_country = df.groupby(['Team', 'Year', 'Season'])\
                        .size()\
                        .reset_index(name='Medal Count')
                        
print(medals_by_country)

medals_by_country.to_csv('medals_by_country.csv', index=False)
