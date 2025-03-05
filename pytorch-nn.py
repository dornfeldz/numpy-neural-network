import torch
import torch.nn as nn
import torch.optim as optim

data = [
    [[0.5556, 0.8471, 0.3636], 1],
    [[0.3333, 0.7647, 0.6364], 0],
    [[0.7778, 0.9412, 0.2727], 1],
    [[0.4444, 0.7059, 0.8182], 0],
    [[1.0, 1.0, 0.1818], 1],
    [[0.2778, 0.6824, 0.9091], 0],
    [[0.6667, 0.8235, 0.4545], 1],
    [[0.2222, 0.5882, 1.0], 0],
    [[0.8889, 0.9176, 0.3273], 1],
    [[0.3889, 0.7294, 0.7273], 0]
]

x = torch.tensor([d[0] for d in data], dtype = torch.float32)
y = torch.tensor([[d[1]] for d in data], dtype = torch.float32)

class LoanApprovalNN(nn.Module):
    def __init__(self):
        super(LoanApprovalNN, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    
model = LoanApprovalNN()
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 2.2)

for epoch in range(30000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
print(epoch, loss.item())

def predict(features):
    with torch.no_grad():
        features_tensor = torch.tensor([features], dtype=torch.float32)
        return model(features_tensor).item()

print(predict([20/90, 250/850, 20/55]))
print(predict([90/90, 850/850, 1/55]))
print(predict([3/90, 20/850, 55/55]))