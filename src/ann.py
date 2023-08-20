import torch
import torch.nn as nn
import torch.optim as optim


class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


def convert_to_tensor(X_train, y_train, X_test, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32).clone().detach()
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).clone().detach()
    X_test = torch.tensor(X_train, dtype=torch.float32).clone().detach()
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32).clone().detach()
    return X_train, y_train, X_test, y_test


def train_ann(model, X_train, y_train, lr=0.001, epochs=100):
    # Defining loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.2f}')

    return model

