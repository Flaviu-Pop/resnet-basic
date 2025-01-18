import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import tqdm
import copy
import time

from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = [7.5, 7.5]

def show_image(img):
    img = img / 2 + 0.5
    img = img.numpy()
    img = img.transpose([1, 2, 0])
    plt.imshow(img)
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- LOADING DATA ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# We set the batch size
batch_size = 16

# We set the training, validation, and test set sizes, respectively.
train_size = 12800
val_size = 1280
test_size = 1280

# We set the path of the (CIFAR10) dataset on the local disk
data_root = "C:\Datasets\CIFAR 10\data\cifar10"

# We do the basic transformations on the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# We download the CIFAR10 dataset (for training) if it does not exist at the location specified by the given path
dataset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform
)

# We split the training set into the train and validation datasets
assert train_size + val_size <= len(dataset), "Trying to sample too many elements" \
                                              "Please lower the train or validation set sizes"

train_set, val_set, _ = torch.utils.data.random_split(dataset,
                                                      [train_size, val_size, len(dataset) - train_size - val_size])

# We download the CIFAR10 dataset (for test) if it does not exist at the location specified by the given path
test_set = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)

# We take test_size elements to working on
test_set, _ = torch.utils.data.random_split(test_set,
                                            [test_size, len(test_set) - test_size])

# We now create the DataLoader's for the train, validation, and test datasets, respectively.
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- THE ARCHITECTURE ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# First we define the ResNet block
class ResNetBlock(nn.Module):
    def conv(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=1,
                         bias=False)

    def __init__(self, in_channels: int, out_channels: int, id: bool = True):
        super().__init__()

        self.id = id
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.downsapling = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding='valid',
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.convolutionalLayer1 = self.conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1)
        self.batchNormLayer1 = nn.BatchNorm2d(num_features=out_channels)

        self.convolutionalLayer2 = self.conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1)
        self.batchNormLayer2 = nn.BatchNorm2d(out_channels)

        self.convolutionalLayer3 = nn.Conv2d(in_channels=out_channels,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding='valid',
                                             bias=False)
        self.batchNormLayer2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.convolutionalLayer1(x)
        out = self.batchNormLayer1(out)
        out = F.relu(out)

        out = self.convolutionalLayer2(out)
        out = self.batchNormLayer2(out)

        if self.id == True:
            out += self.identity(x)
        else:
            out = self.convolutionalLayer3(out)
            out += self.downsapling(x)

        out = F.relu(out)

        return out


# Second, we define the ResNet architecture
class ResNetMultiClassClassification(nn.Module):
    def make_block(self, out_channels: int, block_dim: int = 2):
        layers = []
        id: bool

        if block_dim > 1:
            id = False
        else:
            id = True

        layers.append(ResNetBlock(in_channels=self.in_channels, out_channels=out_channels, id=id))
        self.in_channels = out_channels

        for _ in [block_dim - 1, 1]:
            if _ > 1:
                id = False
            else:
                id = True
            layers.append(ResNetBlock(in_channels=out_channels, out_channels=out_channels, id=id))

        block = nn.Sequential(*layers)

        return block

    def __init__(self):
        super().__init__()

        num_classes = 10
        self.in_channels = 8

        self.convolutionalLayer1 = nn.Conv2d(in_channels=3,
                                             out_channels=self.in_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False)
        self.batchNormLayer1 = nn.BatchNorm2d(self.in_channels)

        self.layer1 = self.make_block(out_channels=8, block_dim=1)
        self.layer2 = self.make_block(out_channels=16, block_dim=2)
        self.layer3 = self.make_block(out_channels=32, block_dim=1)

        self.linearLayer = nn.Linear(1568, num_classes)

    def forward(self, x):
        out = self.convolutionalLayer1(x)
        out = self.batchNormLayer1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)

        out = self.linearLayer(out)

        return out


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- TRAIN ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_accuracy(model, data_loader):
    # Computes the <model>'s accuracy on the <data_loader> dataset

    model = model.to(device)  # Set the <model> to GPU if available
    model.eval()  # Set the model to evaluation mode

    total_correct = 0
    for input, label in tqdm(data_loader):
        input, label = input.to(device), label.to(device)

        output = model(input)
        output_class = output.argmax(1)
        correct = (output_class == label)

        total_correct += correct.sum()

    print(f"\nCorrect Items: {total_correct} --- All Items: {len(data_loader.dataset)}")

    return total_correct / len(data_loader.dataset)


def train(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    print("\n\n\n----------------   The Training Process   ------------------------")

    model = model.to(device)  # Set the model to GPU if available

    best_accuracy = -np.inf
    best_weights = None
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        start_time = time.perf_counter()

        model.train()  # Set the model in training mode

        total_loss = 0
        for input, label in tqdm(train_loader):
            input, label = input.to(device), label.to(device)  # Set the data to GPU if available

            # Forward and Backward passes
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss

        # Take the model('s weights) with the best accuracy
        print(f"\n Computing the Validation Accuracy for Epoch {epoch + 1}:")
        validation_accuracy = compute_accuracy(model, val_loader)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Loss = {total_loss: .3f} ===> Time = {duration: .3f} ===> "
              f"Validation Accuracy = {validation_accuracy: .4f} ===> Best Accuracy = {best_accuracy: .4f} at "
              f"the Epoch {best_epoch} \n")

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    print(f"\n Computing the Test Accuracy for the Best Model")
    test_accuracy = compute_accuracy(model, test_loader)
    print(f"The test accuracy for the model is: {test_accuracy: .4f} --- best model obtained at epoch {best_epoch}")

    # Save the best model on local disk, based on the accuracy of the validation set
    path_best_model = "..\\resnet_cnn_multiclassclassification.pth"
    torch.save(model, path_best_model)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- MAIN ---------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    num_epochs = 10
    resnet = ResNetMultiClassClassification()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9)

    train(resnet, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
          num_epochs=num_epochs, criterion=criterion, optimizer=optimizer)
