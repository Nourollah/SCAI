import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=15, embedding_dim=64)
        self.linear1 = nn.Linear(67, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, atomic_numbers_l, x_3):
        vectors_lu = self.embedding_layer(atomic_numbers_l)
        # concat with x_3
        x = torch.cat((vectors_lu, x_3), dim=-1)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = self.linear3(x)
        x = self.linear4(x).squeeze(-1)
        x = x.sum(axis=-1)
        return x


# load the adam optimizer
def load_optimizer(model, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = {'pos': torch.tensor(
        [[28.2711, 33.3599, 32.7360],
         [29.2658, 31.3207, 32.6588],
         [33.1874, 33.2655, 32.6199],
         [34.4164, 32.3546, 32.6764],
         [31.8915, 32.4476, 32.6568],
         [35.7404, 33.1217, 32.6793],
         [30.6430, 33.3302, 32.6414],
         [36.9646, 32.2198, 32.7081],
         [29.3532, 32.5406, 32.6704],
         [33.2172, 33.8695, 31.7053],
         [33.2072, 33.9597, 33.4688],
         [34.3649, 31.7307, 33.5774],
         [34.3987, 31.6720, 31.8177],
         [31.8787, 31.8192, 33.5564],
         [31.8660, 31.7638, 31.7991],
         [35.7712, 33.7891, 33.5487],
         [35.7924, 33.7584, 31.7880],
         [30.6390, 33.9459, 31.7351],
         [30.6541, 33.9901, 33.5163],
         [37.8797, 32.8201, 32.7094],
         [36.9873, 31.5647, 31.8315],
         [36.9662, 31.5931, 33.6054],
         [27.4311, 32.8541, 32.7656]], device=device),

        'atomic_numbers': torch.tensor([8, 8, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 dtype=torch.long, device=device),
        'energy': torch.tensor(-823.9475059733433, device = device),
        'forces': torch.tensor([[2.3913e-01, -6.8966e-01, -5.0808e-02],
                          [-3.8409e-02,  1.9230e+00, -1.3544e-02],
                          [-4.0261e-02,  7.7610e-03,  4.1796e-02],
                          [8.5530e-02, -2.1338e-01, -2.0676e-03],
                          [3.5616e-02, -2.5226e-01, -2.2612e-02],
                          [-1.0979e-01,  3.4000e-01, -3.1598e-02],
                          [3.7554e-02, -7.1100e-02, -1.7763e-03],
                          [1.5450e-01, -2.1729e-01,  1.0392e-02],
                          [-3.2600e-01, -1.2738e+00,  1.1930e-01],
                          [3.0847e-03, -1.0001e-01,  8.9340e-02],
                          [3.4388e-03, -1.0289e-01, -7.5427e-02],
                          [-1.3983e-02,  1.1311e-01, -8.4776e-02],
                          [-1.5868e-03,  1.1510e-01,  7.6089e-02],
                          [-6.9322e-03,  1.4187e-01, -1.1933e-01],
                          [-1.9607e-02,  1.4240e-01,  1.1098e-01],
                          [-4.6408e-02, -7.8741e-02, -5.7974e-02],
                          [-4.8878e-02, -7.4186e-02,  5.9219e-02],
                          [-7.0732e-02,  2.8147e-02,  1.6528e-01],
                          [-1.0010e-01,  2.2657e-02, -1.8398e-01],
                          [-1.8648e-01, -1.2141e-01,  3.4419e-04],
                          [-2.2751e-02,  1.1878e-01,  1.7670e-01],
                          [-1.6634e-02,  1.0977e-01, -1.7785e-01],
                          [4.8970e-01,  1.3214e-01, -2.7693e-02]], device=device)
        }

# train the model


def train_model(num_epochs=1000):
    model = SimpleCNN().to(device)
    model.train()
    criterion = nn.MSELoss()
    x_3 = data['pos']
    atomic_numbers_l = data['atomic_numbers']
    targets = torch.tensor(data['energy'])
    optimizer = load_optimizer(model)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(atomic_numbers_l, x_3)
        # print(f"Epoch {epoch+1}/{num_epochs} - Outputs: {outputs:.12f}, Targets: {targets}")
        # print(outputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}')


train_model()