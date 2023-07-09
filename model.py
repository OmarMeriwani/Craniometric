import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sex = 0
df = pd.read_excel("Howell.xlsx","Sheet1")
labels_sex = [d[2] for d in df.values]
labels_group = [d[3] for d in df.values if d[2] == 2]
x=[]
if sex == 0:
    x = [d[5:] for d in df.values if d[2] == 2]
else:
    x = [d[5:] for d in df.values ]
x = [y.tolist() for y in x]
if sex == 0:
    x_train, x_test, y_train, y_test = train_test_split(x, labels_group, test_size=0.2)
else:
    x_train, x_test, y_train, y_test = train_test_split(x, labels_sex, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(120,160,180))
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)
