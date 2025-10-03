import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class BasicFeedForward(nn.Module):
    def __init__(self):
        super(BasicFeedForward, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")   ## going to be this one for me

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = BasicFeedForward().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs): ## loop over the dataset multiple times
        total_loss = 0
        correct = 0
        total = 0

        model.train() ## set the model to training mode

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()   ## we don't want to accumulate gradients
            outputs = model(images)
            loss = criterion(outputs, labels) ## computes the loss
            loss.backward() ## backpropagation
            optimizer.step()    ## update the weights

            total_loss += loss.item()
            pred = outputs.argmax(dim=1) ## get the index of the max log-probability
            correct += (pred == labels).sum().item() ## number of correct predictions
            total += labels.size(0) ## total number of labels

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total ## accuracy in percentage
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")


    torch.save(model.state_dict(), "model.pth") ## save the model to a file


    

