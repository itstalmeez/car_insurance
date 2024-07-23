import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
file_path = 'car_insurance.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()
    
# Check for missing values
missing_values = df.isnull().sum()

# Display data types
data_types = df.dtypes

missing_values, data_types
# Check for missing values
missing_values = df.isnull().sum()

# Display data types
data_types = df.dtypes

print("Missing Values:\n", missing_values)
print("\nData Types:\n", data_types)
# Convert categorical features to numerical
df['age'] = df['age'].astype('int')
df['gender'] = df['gender'].astype('int')
df['driving_experience'] = df['driving_experience'].map({'0-9y': 0, '10-19y': 1, '20-29y': 2, '30y+': 3})
df['education'] = df['education'].map({'none': 0, 'high school': 1, 'university': 2})
df['income'] = df['income'].map({'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})
df['vehicle_year'] = df['vehicle_year'].map({'before 2015': 0, 'after 2015': 1})
df['vehicle_type'] = df['vehicle_type'].map({'sedan': 0, 'sports car': 1})

# Drop rows with missing values
df = df.dropna()

features = df.columns.drop(['id', 'outcome'])
X = df[features]
y = df['outcome']

accuracies = []

for feature in features:
    X_feature = X[[feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append((feature, accuracy))

best_feature, best_accuracy = max(accuracies, key=lambda x: x[1])

best_feature_df = pd.DataFrame({
    'best_feature': [best_feature],
    'best_accuracy': [best_accuracy]
})

print(best_feature_df)
