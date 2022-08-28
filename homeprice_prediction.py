import pandas as pd
df = pd.read_csv('homeprices.csv')
# print(df.head())
dummies = pd.get_dummies(df['town'])
# print(dummies.head())
marged = pd.concat([df,dummies],axis='columns')
# print(marged.head())
final = marged.drop(['town','west windsor'],axis='columns')
# print(final.head())

# A graph which shows how price is behaving with respect to area size
import matplotlib.pyplot as plt

plt.scatter(df['area'],df['price'])
plt.show()                                #After closing the graph other result will be generated

#Building a machine learning model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = final.drop('price',axis='columns')
# print(X.head())
y = final['price']
# print(y.head())
model.fit(X,y)

# When the area is 5000 sqft. and the town is West windsor

result = model.predict([[5000,0,0]])

print('The predicted price will be ',int(result),'Rs.')

score_model = model.score(X,y)

print('The score of the model is : ',score_model*100,'%')

