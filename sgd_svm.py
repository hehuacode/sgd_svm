#!/usr/bin/python3

#Homework3 by Hehua Chi


import numpy as np
import argparse

def read_adult(path):
    # Read the adult data by file path   
    file_data = open(path)
    data_lines = file_data.readlines()
    
    # initialize the array to store feature values and target values
    x = np.zeros((len(data_lines),123))
    y = np.zeros(len(data_lines))
    
    #content the zeros array with the loaded data
    for count, dataline in enumerate(data_lines):
        line = dataline.split()
        y[count] = int(line[0])
        
        # based on the number before :, we assign the position as 1
        for j in range(1, len(line)):
            column = int(line[j].split(":")[0])-1
            x[count,column] =1
    return x, y


# use the SGD to optimize the w and bias b, based on the SGD of SVM algorithm, there are six input
# maximize the objective function is (1/2)*(w^T)w + C Sum(1..N) max(0, 1 - yn(w^T x^(n) +b)).
# based on the Formulation: Your optimization problem for the adult dataset should match (13) on pp. 20 of the lecture note.
# sgd function is to calulate the gradient of this optimizition problem
def sgd(xn, yn, N, w_vector, b, capacity):
	if 1-yn*(np.dot(w_vector,xn) + b)>0:
		dw = (1/N)*w_vector - capacity*yn*xn
		db = - capacity*yn
	else:
		dw = w_vector/N
		db = 0
	return dw,db

    
#The learning rate should be 0.1 across your training process.
def sgd_svm(data_x, data_y, epochs, capacity, learning_rate = 0.1):

    #initialize the w and b 
	w_vector = np.zeros(123)
	b = 0
    #calculate the size of x
	N = data_x.shape[0]
    
	for i in range(epochs):
		for n in range(N):
			dw, db = sgd(data_x[n], data_y[n], N, w_vector, b, capacity)
			w_vector -= learning_rate * dw
			b -= learning_rate * db
	return w_vector,b


def acc(data_x, data_y, w, b):
  #calculate the total number of data  
	total_num = len(data_x)
  #initialize the correct number
	correct_num = 0
  # use the trained w and b, calculate the sign() to track the correctness
	for i in range(total_num):
		if (np.dot(data_x[i],w)+b)*data_y[i]>0:
			correct_num+=1
  # calculate the accuracy
	return correct_num/total_num



# Use the argparse to format the tranfered parameters
parser = argparse.ArgumentParser(description='chi_hehua_hw3')

#add the argument epochs
parser.add_argument('--epochs')

#add the argument capacity
parser.add_argument('--capacity')

epochs = 1
capacity = 0.868

args = parser.parse_args()
if args.epochs:
	epochs = int(args.epochs)
if args.capacity:
	capacity = float(args.capacity)

 
 
train_data = read_adult('/u/cs246/data/adult/a7a.train')
dev_data = read_adult('/u/cs246/data/adult/a7a.dev')
test_data = read_adult('/u/cs246/data/adult/a7a.test')

if __name__ == "__main__":
  # use sgd_svm algorithm to calculate the w and b
	w,b = sgd_svm(train_data[0], train_data[1], epochs, capacity, learning_rate = 0.1 )
  # use the generated w and b to calcuate the accuracy for train, test, dev
	train_accuracy = acc(train_data[0], train_data[1], w, b)
	dev_accuracy = acc(dev_data[0], dev_data[1], w, b)    
	test_accuracy = acc(test_data[0], test_data[1], w, b)
  # print the result
	print("EPOCHS: ", epochs)
	print("CAPACITY: ", capacity)
	print("TRAINING_ACCURACY: %.8f" %train_accuracy)
	print("TEST_ACCURACY: %.8f" %test_accuracy)
	print("DEV_ACCURACY: %.8f" %dev_accuracy)
	print("FINAL_SVM: ", [b] + w.tolist())
