# sgd_svm
Implement SGD for SVM for the adult income dataset. Experiment with performance as a function of the capacity parameter C
************ Algorithm *****
My algorithm optimization problem for the adult dataset match (13) on pp. 20 of the lecture note.


************ Instructions ***

The python program can be run from the command on the cycle machine:
./chi_hehua_hw3.py --epochs 1 --capacity 0.868

In order to generate the plot.pny fast, I store all the generated accuracies into vectors test_acc() and dev_acc()
use the command: ./plot.py


************ Results *******
The result shows the accuracy will change with capacity c. 


************ Your interpretation **** (Attention: important!)

From the plot.png, we can see that the accuracy will be highest when capacity is between 0.01 and 1. And if capacity is bigger than 10, the accuracy will be not stable and not very good. So for SGD OF SVM, we should select capacity c around 1.

************ References ************
1. Christopher M. Bishop, Pattern Recognition and Machine Learning
2. Lecture Notes
