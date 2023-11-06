import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=transform, train=True)
train_cifar_dataloader = DataLoader(train_cifar_dataset, batch_size=5000, shuffle=True)

test_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=transform, train=False)
test_cifar_dataloader = DataLoader(test_cifar_dataset, batch_size=5000, shuffle=True)

class CIFARPredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x

model = CIFARPredictorPerceptron(input_size=3072, hidden_size=180, output_size=100)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):
    correct_guess_train = 0
    error_train = 0
    for x, y in train_cifar_dataloader:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        zero_tensor = torch.zeros_like(prediction)
        loss_train = loss_fn(prediction, y)
        error_train += loss_train

        loss_train.backward()
        optimizer.step()

        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess_train += (predicted_indices == y).float().sum()

    writer.add_scalar('Loss/train', error_train/len(train_cifar_dataset), epoch)
    writer.add_scalar('Accuracy/train', correct_guess_train / len(train_cifar_dataset), epoch)
    print('Train loss ' + str(error_train/len(train_cifar_dataset)))
    print('Train accuracy ' + str(correct_guess_train / len(train_cifar_dataset)))


    correct_guess_test = 0
    error_test = 0
    for x,y in test_cifar_dataloader:
        model.eval()
        prediction = model(x)
        loss_test = loss_fn(prediction, y)
        error_test += loss_test

        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess_test += (predicted_indices == y).float().sum()

    writer.add_scalar('Loss/test', error_test/len(test_cifar_dataset), epoch)
    writer.add_scalar('Accuracy/test', correct_guess_test/len(test_cifar_dataset), epoch)
    print('Test loss ' + str(error_test / len(test_cifar_dataset)))
    print('Test accuracy ' + str(correct_guess_test / len(test_cifar_dataset)))



