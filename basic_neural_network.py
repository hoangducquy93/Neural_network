import numpy as np 

#tao dataset 
training_set_input = np.array([[0,0,0],
                              [1,0,1],
                              [1,1,1],
                              [1,1,0],
                              [0,0,1],
                              [0,1,0]])
training_set_output = np.array([[0,1,1,1,0,0]]).T 

np.random.seed(1)
#tao ham sigmoid

def sigmoid(x):
	y = 1/(1+np.exp(-x))
	return y

def sigmoid_derivative(x):
	y = x*(1-x)
	return y
'''
print(training_set_input)
print(training_set_output)
'''
#tao gia tri random tu (-a,b) theo cong thuc b-a * np.random.random(size) + a
synaptic_weights = 2* np.random.random((3,1)) - 1
print('synaptic_weights ban dau:')
print(synaptic_weights)

for iteration in range(10000):
	input_layer = training_set_input

	output = sigmoid(np.dot(input_layer,synaptic_weights))

	error = training_set_output - output


	adjustment = np.dot(input_layer.T,error*sigmoid_derivative(output))

	#cap nhat synaptic_weights
	synaptic_weights += adjustment

print('adjustment:',adjustment)
print('sypnaptic_weights sau update: ',synaptic_weights)

think = sigmoid(np.dot(np.array([0,1,1]),synaptic_weights))
print(think)


