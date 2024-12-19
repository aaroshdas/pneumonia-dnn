import numpy as np
import pickle

data = np.load('pneumoniamnist.npz')

# print(data.files)

train_inp = data['train_images']
train_sol =data['train_labels']

train_set = []

for i in range(len(train_inp)):
    solution = np.zeros((2, 1))
    if(train_sol[i,0] == 0):
        solution[0,0] = 1
    else:
        solution[1,0] = 1 #non pneumonia
    allNums = []
    img = train_inp[i]
    for k in img:
        for num in k:
            allNums.append(num)
    finalInput = np.zeros((784, 1))
    for i in range(len(allNums)):
            finalInput[i, 0] = int(allNums[i])/255
    train_set.append((finalInput,solution))

train_inp = data['train_images']
train_sol =data['train_labels']

train_set = []

for i in range(len(train_inp)):
    solution = np.zeros((2, 1))
    if(train_sol[i,0] == 0):
        solution[0,0] = 1
    else:
        solution[1,0] = 1 #non pneumonia
    allNums = []
    img = train_inp[i]
    for k in img:
        for num in k:
            allNums.append(num)
    finalInput = np.zeros((784, 1))
    #distortions
    # for i in range(len(allNums)):
    #        finalInput[i+1, 0] = int(allNums[i])/255
    # for i in range(len(allNums)):
    #        finalInput[i-1, 0] = int(allNums[i])/255
    
    for i in range(len(allNums)):
            finalInput[i, 0] = int(allNums[i])/255
    train_set.append((finalInput,solution))

test_inp = data['test_images']
test_sol =data['test_labels']

test_set = []

for i in range(len(test_inp)):
    solution = np.zeros((2, 1))
    if(test_sol[i,0] == 0):
        solution[0,0] = 1
    else:
        solution[1,0] = 1 #non pneumonia
    allNums = []
    img = test_inp[i]
    for k in img:
        for num in k:
            allNums.append(num)
    finalInput = np.zeros((784, 1))
    for i in range(len(allNums)):
            finalInput[i, 0] = int(allNums[i])/255
    test_set.append((finalInput,solution))

#1 = pnemonia 
#0 = none 9:1 ratio


def sigmoid(num):
    return(1/(1+np.exp(-num)))
sigmoid_vectorized = np.vectorize(sigmoid)


def activationFuncDerivative(activationFunc,x):
    return activationFunc(x)*(1-activationFunc(x))

def test(inputs, w, b, activationFunc):
    correct = 0
    pCorrect= 0
    for inp in inputs: 
        As ={}
        As[0] = inp[0]
        dots = {}        
        for layer in range(1, len(w)): 
            dots[layer] = (w[layer]@As[layer-1])+b[layer]
            As[layer] = activationFunc(dots[layer])
        max_i = 0
        for i in range(len(As[len(w)-1])):
            if(As[len(w)-1][i, 0] > As[len(w)-1][max_i, 0]):
                max_i = i
        if(inp[1][max_i, 0] == 1):
            correct +=1
            if(max_i == 0):
                pCorrect+=1
    print(correct) #raw correct points /624
    print(pCorrect) #pneumonia correct points /624
    return correct/len(inputs)
       

def back_propagation(inputs, w, b, activationFunc, learningRate, epochs):
    for epoch in range(epochs):     
        for ind, inp in enumerate(inputs): 
            if(ind%1000 == 0):
                print(ind)
            As ={}
            As[0] = inp[0]
            dots = {}        
            for layer in range(1, len(w)):
                dots[layer] = (w[layer]@As[layer-1])+b[layer]
                As[layer] = activationFunc(dots[layer])
            deltas= {}
            deltas[len(w)-1] = activationFuncDerivative(activationFunc, dots[len(w)-1]) * (inp[1]-As[len(w)-1])
            
            for layer in range(len(w)-2, 0,-1):
                deltas[layer] = activationFuncDerivative(activationFunc, dots[layer]) *(np.transpose(w[layer+1])@deltas[layer+1])
            for layer in range(1, len(w)):
                b[layer] = b[layer]+learningRate*deltas[layer]
                w[layer] = w[layer]+learningRate*deltas[layer] *np.transpose(As[layer-1])
        with open("w_b.pkl", "wb") as f:
            pickle.dump((w1,b1), f)
        print("w/b saved")
        print(str(test(test_set, w1, b1, sigmoid)) + "% \n")
    return (w,b)

def create_rand_values(dimensions):
    weights= [None]
    biases = [None]
    for i in range(1,len(dimensions)):
        weights.append(2*np.random.rand(dimensions[i],dimensions[i-1]) - 1)
        biases.append(2*np.random.rand(dimensions[i],1)-1)
    return weights, biases

#test_set 624
#train_set 4706

# w1, b1 = create_rand_values([784, 300,100, 2]) gen random w/b
 
with open("w_b.pkl", "rb") as f:
    w1,b1 = pickle.load(f)
print(str(test(test_set, w1, b1, sigmoid)) + "% \n")
w1, b1 = back_propagation(train_set, w1, b1, sigmoid, 0.01, 5)

