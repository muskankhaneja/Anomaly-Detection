import torch
import torch.nn as nn
import torch.optim as optim
import mlflow


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, data, lr=0.001, epochs=20):
    # Defining loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        avg_loss = running_loss / len(data)
        mlflow.log_metric("loss", avg_loss, step=epoch)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.2f}')

    mlflow.pytorch.log_model(model, "models")

    return


def calculate_reconstruction_error(model, data):
    with torch.no_grad():
        reconstructions = model(data)
        mse_loss = nn.MSELoss(reduction='none')
        errors = mse_loss(reconstructions, X_train).mean(dim=1)
    return errors
