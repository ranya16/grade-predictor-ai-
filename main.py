# AI model by Ranya - Algerian student
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {'hours': [2,4,6,8,10], 'grades': [10,12,14,16,18]}
df = pd.DataFrame(data)

model = LinearRegression()
model.fit(df[['hours']], df['grades'])

# Try to predict for 7 hours
prediction = model.predict([[7]])
print(f"Predicted grade for 7h study: {prediction[0]:.1f}/20")
