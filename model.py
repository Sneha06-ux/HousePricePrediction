import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df=pd.read_csv("house_price_dataset_clean.csv")

X=df[['Size (sq. ft.)','Bedrooms',"Bathrooms","Age of House (Years)","Distance to City Center (Miles)"]]
y=df['Price']

model=LinearRegression()

model.fit(X,y)
pickle.dump(model,open("trained_model.pkl","wb"))
#print("model.predict([[2247.97652292,3,4,20,28.87643]])")
