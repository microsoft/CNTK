
import numpy as np

def create_empty_array_of_shape(shape):
    if shape: return [create_empty_array_of_shape(shape[1:]) for i in range(shape[0])]

def create_array(first_dim = 5,second_dim = 2,third_dim=2):

    array = create_empty_array_of_shape([1,first_dim,second_dim,third_dim])
    #array = [[[[] for i in range(third_dim)]for i in range(second_dim)]for i in range(first_dim)]# [[[.0]*third_dim]*second_dim]*first_dim
    array = array[0]
    print (array)
    for x in range(first_dim):
        print("x=" + str(x))
        for y in range(second_dim):
            print("y=" + str(y))
            for z in range(third_dim):
                print("z=" + str(z))
                array[x][y][z] =  (x*100+y)*100+z  #str(x)+"_"+str(y)+"_"+str(z)
                # print(array)

    # array = np.asarray()

    return array

