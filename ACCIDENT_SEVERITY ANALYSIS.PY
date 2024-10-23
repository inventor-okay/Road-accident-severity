import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('accidents.csv')

# Step 2: Check the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 3: Preprocess the data (encoding categorical variables)
label_encoder = LabelEncoder()
data['weather_conditions'] = label_encoder.fit_transform(data['weather_conditions'])
data['road_type'] = label_encoder.fit_transform(data['road_type'])

# Step 4: Define the dependent and independent variables
X = data[['speed', 'weather_conditions', 'time_of_day', 'road_type']]  # Independent variables
y = data['accident_severity']  # Dependent variable (target)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create the Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 9: Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Step 10: Load the saved model and predict for new hypothetical data
# Example: speed=70, weather_conditions=2 (encoded), time_of_day=15, road_type=1 (encoded)
model = joblib.load('accident_severity_model.pkl')
new_data = [[70, 2, 15, 1]]  # Hypothetical input
predicted_severity = model.predict(new_data)
print(f'\nPredicted Accident Severity for new data: {predicted_severity[0]}')

# Step 11: Visualization (Optional)
sns.scatterplot(x=data['speed'], y=data['accident_severity'])
plt.title('Speed vs. Accident Severity')
plt.show()
