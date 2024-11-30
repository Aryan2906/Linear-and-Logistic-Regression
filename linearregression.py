import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv("Dataset path learning floor matrix task.csv")
data.pop("ID")
data.pop("Gender")
data.pop("Counterbalancing floor matrix task")
dataclean=data.dropna()
df=pd.get_dummies(dataclean)
x=df[["Peabody","Raven","SAQ","PMA-SR-K1","GPT_total","WM_matr_sequential","WM_matr_simultaneous","Floor Matrix Map","Floor Matrix Obs"]]
y=df[["Group_Down","Group_TD"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test,y_test))
