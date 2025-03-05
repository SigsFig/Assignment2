#-------------------------------------------------------------------------
# AUTHOR: Nicholas Hoang
# FILENAME: naive_bayes.py
# SPECIFICATION: Uses the Naive Bayes algorithm to predict whether to play tennis or not based off forecast data. Outputs predictions with 0.75 confidence or greater
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
import csv
from sklearn.naive_bayes import GaussianNB

#Reading the training data in a csv file
#--> add your Python code here
X = []
Y = []

feature_mapping = {
        'Sunny': 1, 'Overcast': 2, 'Rain': 3,
        'Hot': 1, 'Mild': 2, 'Cool': 3,
        'High': 1, 'Normal': 2,
        'Weak': 1, 'Strong': 2
}
class_mapping = {'Yes': 1, 'No': 2}

with open('weather_training.csv', 'r') as file:
    reader = csv.reader(file)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
    next(reader)
    
    for row in reader:
        X.append([feature_mapping[row[1]], feature_mapping[row[2]], feature_mapping[row[3]], feature_mapping[row[4]]])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
        Y.append(class_mapping[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

test_samples = []
#Printing the header os the solution
#--> add your Python codprinte here
print("Day        Outlook      Temperature  Humidity   Wind     PlayTennis  Confidence")
#Reading the test data in a csv file
#--> add your Python code here
with open('weather_test.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        #Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
        #--> add your Python code here
        prediction = clf.predict_proba([[feature_mapping[row[1]], feature_mapping[row[2]], feature_mapping[row[3]], feature_mapping[row[4]]]])[0]
        if max(prediction[0], prediction[1]) >= 0.75:
            print(f"{row[0]:<10} {row[1]:<12} {row[2]:<12} {row[3]:<10} {row[4]:<8} {('Yes' if prediction[0] > prediction[1] else 'No'):<11} {round(max(prediction[0], prediction[1]),2)}")