
# coding: utf-8

# In[8]:


#imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from random import random
from random import seed
from math import exp


# In[9]:


# No. of data points
n_samples = 200

# No. of features (dimensions of the data)
n_features = 4

# No. of redundent features (linear combinations of other features)
n_redundant = 1

# No. of classes
n_classes = 2


# In[10]:


X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                            n_redundant=n_redundant, n_classes=n_classes)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['label'] = y
df.head()


# In[11]:


#df.to_csv("dataset1.csv")


# In[12]:


#reading the dataset 
#dataset was generated from make-dataset.pynb
df=pd.read_csv('dataset1.csv',index_col=0)
df.head()


# In[13]:


#initializing the network for weights
def initialize_network(n_inputs, n_hidden, n_outputs):
    network=list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# In[14]:


#activation function
def activate(weights, inputs):
    activation=weights[-1]
    for i in range(len(weights)-1):
        activation+=weights[i]*inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# In[15]:


#forward propogation
def forward_propagate(network,raw):
    inputs=raw
    for layer in network:
        new_inputs=[]
        for neuron in layer:
            activation=activate(neuron['weights'], inputs)
            neuron['output']=transfer(activation)
            new_inputs.append(neuron['output'])
        inputs=new_inputs
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


# In[16]:


#backward propogation to learn
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# In[17]:


#update weights on training
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs=row[:-1]
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=l_rate*neuron['delta']*inputs[j]
            neuron['weights'][-1]+=l_rate*neuron['delta']


# In[18]:


#training the network
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# In[19]:


#predicting function
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# In[20]:


dataset=np.array(df[:])
dataset


# In[21]:


n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
print(n_inputs,n_outputs)


# In[22]:


#splitting into test and train datset
train_dataset=dataset[:150]
test_dataset=dataset[150:]


# In[23]:


#feeding the datset into the network
network=initialize_network(n_inputs,1,n_outputs)
train_network(network, train_dataset, 0.5, 100, n_outputs)


# In[24]:


#learned weights of the network
for layer in network:
    print(layer)


# In[25]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[26]:


#applying on training dataset
y_train=[]
pred=[]
for row in train_dataset:
    prediction = predict(network, row)
    y_train.append(int(row[-1]))
    pred.append(prediction)


# In[27]:


print("Accuracy: ",accuracy_score(y_train,pred))
print("Confusion Matrix: ",confusion_matrix(y_train,pred))
print("Precision: ",precision_score(y_train, pred))
print("recall: ",recall_score(y_train, pred))


# In[28]:


#applying on testing dataset
y_test=[]
pred=[]
for row in test_dataset:
    prediction = predict(network, row)
    y_test.append(row[-1])
    pred.append(prediction)


# In[29]:


print("Accuracy: ",accuracy_score(y_test,pred))
print("Confusion Matrix: ",confusion_matrix(y_test,pred))
print("Precision: ",precision_score(y_test, pred))
print("recall: ",recall_score(y_test, pred))



