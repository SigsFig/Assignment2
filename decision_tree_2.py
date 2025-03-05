#-------------------------------------------------------------------------
# AUTHOR: Nicholas Hoang
# FILENAME: decision_tree_2.py
# SPECIFICATION: Trains decision tree models based off 3 eye contact lens datasets of varying sizes, and tests them on a test dataset
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

feature_mapping = {
        'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3,
        'Myope': 1, 'Hypermetrope': 2,
        'Yes': 1, 'No': 2,
        'Normal': 1, 'Reduced': 2
}
class_mapping = {'Yes': 1, 'No': 2}

for ds in dataSets:

    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
                X.append([feature_mapping[row[0]], feature_mapping[row[1]], feature_mapping[row[2]], feature_mapping[row[3]]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
                Y.append(class_mapping[row[4]])

    #Loop your training and test tasks 10 times here
    correct = 0
    incorrect = 0

    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []

        with open('contact_lens_test.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                dbTest.append([feature_mapping[row[0]], feature_mapping[row[1]], feature_mapping[row[2]], feature_mapping[row[3]], feature_mapping[row[4]]])
        
        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if class_predicted == data[4]:
                correct += 1
            else:
                incorrect += 1

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    accuracy = correct / (correct + incorrect)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f'Final accuracy when training on {ds}: {accuracy:.2f}')