import numpy as np

# Load the data files
data_file1 = 'EOLAS_test_set_real.npy'
data_file2 = 'predicted_strengths_EOLAS_real.npy'

# Load data from files
real = np.loadtxt(data_file1)
predict = np.loadtxt(data_file2)
x = (real[:, 2])

total_average = 0
average = 0
search = 0
MSE = 0
total_MSE=0
MAE = 0
total_MAE = 0





for i in range(0,72):
    sum_data1 = x[i]
    sum_data2 = predict[i]
    if((sum_data1 != 0.0)) and ((sum_data2 != 0.0)):
        average = np.abs(sum_data1 - sum_data2) / np.abs(sum_data1) * 100
        MSE = (sum_data1-sum_data2)**2
        MAE = np.abs(sum_data1-sum_data2)
        total_average += average
        total_MSE +=MSE
        total_MAE +=MAE
        if(MAE <= 28):
            search+=1
    print("Sum1:", sum_data1)
    print("Sum2:", sum_data2)
    print("Error %:", average)
    print("MSE:", MSE)
    print("MAE:", MAE)

print (search)
print("Average Error %: ",(total_average)/72)
print("MSE: ",(total_MSE)/72)
print("MAE: ",(total_MAE)/72)
print("RMSE: ",((total_MSE)/72)**0.5)
print("Count: ",search)

