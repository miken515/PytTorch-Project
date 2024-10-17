import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

'''
Below is the class for pre-built neural networks with Pytorch
We initialize it with multiple layers such as a convolution layers and linear layers
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution layer with in channels for input, output channels produced by convolution, and kernel size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Linear layer with in and out samples
        self.l1 = nn.Linear(320, 50)
        self.l2 = nn.Linear(50, 10)

    # This foward function is how the input goes through the layers
    def fwd(self, val):
        # Applies linear unit (a math function reLU(X) = (x) = max(0,x)
        val = nn.functional.relu(nn.functional.max_pool2d(self.conv1(val), 2))
        val = nn.functional.relu(nn.functional.max_pool2d(self.conv2(val), 2))

        # Return tensor in a different shap
        val = val.view(-1, 320)
        val = nn.functional.relu(self.l1(val))
        val = self.l2(val)
        return nn.functional.log_softmax(val, dim=1)

'''
Below is a training function where we train the network. We pass multiple inputs of data to train
It computes losses of things it did not catch. Also, it does forward and backwards processing to train
'''
def training(model, device, trainingLoader, optimizer, epoch):
    # Informing the pre-built model that we are going to train it
    model.train()
    for batch, (data, target) in enumerate(trainingLoader): # Building a dictionary
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(data), len(trainingLoader.dataset),
                       100 * batch / len(trainingLoader), loss.item()))

'''
Just like the training function, this is where we test how accurate the work can be
'''
def testing(model, device, testingLoader):
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testingLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += nn.functional.nll_loss(output, target, reduction = 'sum').item()
            prediction = output.argmax(dim = 1, keepdim = True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    testLoss /= len(testingLoader.dataset)

    print('Avg Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(testLoss, correct, len(testingLoader.dataset),
        100. * correct / len(testingLoader.dataset)))


'''
Below is where we bring the data so the network can analyze. We will be using the MNIST dataset, which is 
handwritten digits that is used to train/test images
'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(.13, .31)])

trainingDataSet = datasets.MNIST('../data', train=True, download=True, transform=transform)
testDataSet = datasets.MNIST('../data', train=False, download=True, transform=transform)

testingLoader = torch.utils.data.DataLoader(trainingDataSet, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1000, shuffle=True)

# Device will prioritize gpu power but if not, we will use the CPU power of the computer size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializing optimizer function and model
model = Net().to(device) # moves it to a specific device.. either gpu or cpu

# the optimizer is using an SDG algorithm to update model parameters
optimizer = optim.SGD(model.parameters(), lr = .01, momentum = .5)

for epoch in range(10):
    training(model, device, testingLoader, optimizer, epoch)
    testing(model, device, testLoader)

    trainingLosses, trainingCorrect, trainingTotal = training(model, device, testLoader, optimizer, epoch)
    testingAccuracy = 100. * trainingCorrect / trainingTotal
    testingLosses, testingCorrect, testingTotal = testing(model, device, testLoader, test_loss=True)
    testingAccuracy = 100. * testingCorrect / testingTotal

    print('Epoch: [{}/{}]'.format(epoch + 1, 10))
    print('Avg training loss: {:.4f}'.format(trainingLosses[-1]))
    print('Accuracy: {:.2f}%'.format(testingAccuracy))

