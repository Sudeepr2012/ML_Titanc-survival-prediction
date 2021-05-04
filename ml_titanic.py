
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

test_data=pd.read_csv("https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv")
train_data=pd.read_csv("https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv")

x_1=train_data.drop(["PassengerId","Survived","Name","Ticket","Cabin","Embarked"],axis=1)
x_1=x_1.fillna(0)
y_1=train_data["Survived"]
y_1=y_1.fillna(0)
x_1,y_1

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
cf= make_column_transformer(
    (MinMaxScaler(),["Pclass", "Age", "SibSp", "Parch","Fare"]),
    (OneHotEncoder(handle_unknown="ignore"),["Sex"])
)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x_1,y_1, random_state=42)
cf.fit(x_train)
x_train_normal=cf.transform(x_train)
x_test_normal=cf.transform(x_test)
tf.random.set_seed(42)
model_titanic=tf.keras.Sequential([
      tf.keras.layers.Dense(10, input_shape=(7,), activation="relu"),
      tf.keras.layers.Dense(600, activation="tanh"),
      tf.keras.layers.Dense(200, activation="tanh"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1, activation="sigmoid")
])

model_titanic.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(lr=3e-4, decay=1e-6),
                      metrics=["accuracy"])
history=model_titanic.fit(x_train_normal,y_train, epochs=500, verbose=0)

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
model_titanic.evaluate(x_test_normal, y_test)

test_data=test_data.fillna(0)
test_data

Id=test_data["PassengerId"].values
test_data=test_data.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)
test_data_normal=cf.transform(test_data)
test_data_normal
predictions= model_titanic.predict(test_data_normal)
pred=[1 if predictions[i]>0.25 else 0 for i in range(len(test_data_normal))]
pred
table=pd.DataFrame()
data=pd.read_csv("https://raw.githubusercontent.com/Ajiteshrock/TITANIC-SURVIVAL-PREDICTION/master/titanic/gender_submission.csv")
data.head()

table['PassengerId']=Id
table['Survived']=pred
table.head()
