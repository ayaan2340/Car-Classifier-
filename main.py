import numpy as numpy
import sklearn as sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

# Read the input lines from car.data
f = open('car.data','r')
lines = f.readlines()
f.close()

# Create dictionaries to convert input words into class numbers
values1 = {'vhigh': 1,'high': 2,'med': 3,'low': 4}
values2 = {'vhigh': 1,'high': 2,'med': 3,'low': 4}
values3 = {'2': 1,'3': 2,'4': 3,'5more': 4}
values4 = {'2': 1,'4': 2,'more': 3}
values5 = {'small':  1,'med': 2,'big': 3}
values6 = {'low': 1,'med': 2,'high': 3}
values7 = {'unacc\n': 1,'acc\n': 2,'good\n': 3,'vgood\n': 4}
values = [values1,values2,values3,values4,values5,values6, values7]

# Feed each data sample into X and Y using the dictionaries to convert the words into classes
X = numpy.array([])
Y = numpy.array([])
for i in range(len(lines)):
    split = lines[i].split(',')
    for j in range(len(split)):
        split[j] = values[j][split[j]]
    X = numpy.append(X, split[0:6])
    Y = numpy.append(Y, split[6])
X = X.reshape(len(lines),6)

# Splits the data into training and testing sets with 10% of the data being used for testing
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Finds the best k value for the KNN algorithm
highest = 0
for i in range(1, 20):
    neighbors = KNeighborsClassifier(n_neighbors=i)
    neighbors.fit(xtrain, ytrain)
    score = neighbors.score(xtest, ytest)
    print(str(score) + "% " + str(i) + " neighbors")
    if score > highest:
        highest = score
        best = i

print(str(best) + " neighbors is the best number of neighbors")
neighbors = KNeighborsClassifier(n_neighbors=best)
neighbors.fit(xtrain, ytrain)

# Asks for user input to predict the class of a new data sample
while True:
    print("Enter a car's details or type 'exit' to exit: ")
    inputBuying = input("Buying Cost (vhigh, high, med, low): ")
    if inputBuying == "exit":
        break
    inputMaint = input("Maintinence (vhigh, high, med, low): ")
    inputDoors = input("Number of Doors (2, 4, 5more): ")
    inputPersons = input("Number of Persons (2, 4, more): ")
    inputLug_boot = input("Trunk (small, med, big): ")
    inputSafety = input("Safety Rating (low, med, high): ")
    inputX = [[values1[inputBuying], values2[inputMaint], values3[inputDoors], values4[inputPersons], values5[inputLug_boot], values6[inputSafety]]]

    print(str(neighbors.predict(inputX)) + " is the predicted class with 4 being the highest and 1 being the lowest")
    print("")