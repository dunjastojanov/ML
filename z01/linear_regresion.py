import numpy as np
from random import randint
import sys

def load_file(file_path):
    f = open(file_path, "r")
    f.readline()
    lines = f.readlines()
    
    values = {}
    for line in lines:
        y, x = line.strip().split(',')
        values[float(x)] = float(y)
        
    x_values = sorted(values.keys())
    y_values = [values[x] for x in x_values] 
    f.close()
    return x_values, y_values


def rmse(y_values, calculated_values):
    suma = 0
    for expected, actual in zip(calculated_values, y_values):
        suma += (actual-expected) ** 2
    
    return np.sqrt(suma/len(y_values))

def predict(x, t1, t0):
    return x*t1 + t0

def filter_data(x_train, y_train):
    filtered_x = []
    filtered_y = []
    for x, y in zip(x_train, y_train):
        if not (y>900 or (x>5 and y<200) or (x>3 and y<20)):
            filtered_x.append(x)
            filtered_y.append(y)
    
    return np.array(filtered_x), np.array(filtered_y)

def normal_equation(x_values, y_values):
    x_len = len(x_values)
    x_matrix = np.column_stack((np.ones((x_len, 1)), x_values))
    y_col = y_values.reshape(-1, 1)
    y_col = y_col.astype(float)
    x_transpose = np.transpose(x_matrix)
    x_matrix = x_matrix.astype(float)
    x_transpose = x_transpose.astype(float)
    inverse = np.linalg.inv(np.dot(x_transpose, x_matrix))
    teta = np.dot(np.dot(inverse, x_transpose), y_col)
    return teta

def get_rmse(x_train, y_train, x_test, y_test, fit_method):
    t0, t1 = fit_method(x_train, y_train)
    calculated_values = [predict(x_t, t1, t0)[0] for x_t in x_test]
    return rmse(y_test, calculated_values)

def linear_regresion(train_file, test_file):
    x_train, y_train = load_file(train_file)
    x_test, y_test = load_file(test_file)
    filtered_x, filtered_y = filter_data(x_train, y_train)
	
    nq = get_rmse(filtered_x, filtered_y, x_test, y_test, normal_equation)
    print(nq)


if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    linear_regresion(train_file, test_file)



