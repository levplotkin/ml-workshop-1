# MNIST with PyTorch 

supervised learning: labeled data > model > target
classification: result is prediction vector

activation function - (not linear) enables neurons in next layer
loss function   
gradient decrease method
training: model(training set) > target + actual target > loss function > backpropagation > weights correction 
validation  : model(validation set) > target + actual target > loss function  > check model for overwriting/underwriting
training + validation = epoch
test  : model(test set) > target + actual target 

data preparation

raw data >> transformation >> split to test and training (80%/20%)
train set >> split train ti train and validation    

NN model setup:
hyperparameters:
    layers number
    neurons number in layers
    learning rate
    train/val/test ratio
    activation function
    epoch number
    loss function
    batch size
layers configuration
    Linear(input, output)
activation function: 
    transforms outputs from one layer to next  
    ReLU, Sigmoid, Tangh, Softmax
loss function: 
    mesures the error of NN
    we want to minimize 
    MSE, MAE, cross-entropy, binary cross-entropy, CTCLoss  
Gradient decent optimizer
    Adam, RMSprp, SGD
Metrics: rate for measuring model quality
    Accuracy, 
