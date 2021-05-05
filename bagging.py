import random
import math
import mysklearn.myevaluation as myevaluation
import mysklearn.myutils as myutils
import mysklearn.myclassifiers as myclassifiers
import mysklearn.mypytable as mypytable

def random_forest(X,y,N,M,F):
    #step 1
    data = myevaluation.train_test_split(X, y, .33)
    print("data split")
    remainder_x = data[0] 
    remainder_y = data[2]
    validation_x = data[1]
    validation_y = data[3]

    forest = myclassifiers.MyRandomForestGenerator()
    print("begining fit")
    forest.fit(remainder_x,remainder_y,N,M,F)
    print("forest fit")
    answers = forest.predict(validation_x)
    accuracy = 0
    for x in range(len(answers)):
        if validation_y[x]-1.0 <= answers[x] and answers[x] <= validation_y[x]+1.0:
            accuracy += 1
    print(accuracy/len(answers))


header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
    "lang": ["R", "Python", "Java"],
    "tweets": ["yes", "no"], 
    "phd": ["yes", "no"]}
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
#random_forest(X, y, 20, 7, 2)

table = mypytable.MyPyTable()
table.load_from_file("input_data/winemag-data_first150k.csv")
print(table.column_names)
country = table.get_column("country")
province = table.get_column("province")
price = table.get_column("price")
variety = table.get_column("variety")

wine_data = []
for x in range(len(country)):
    entry = []
    entry.append(country[x])
    entry.append(province[x])
    entry.append(price[x])
    entry.append(variety[x])
    wine_data.append(entry)
wine_data = wine_data[:10000]

scores = table.get_column("points")
random_forest(wine_data, scores, 100, 10, 2)