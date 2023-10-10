import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch import Tensor
import wandb


class WM(nn.Module):
    def __init__(self, in_features: int, out_features: int, expert_class: nn.Module, n_experts: int, rho=1.0, eps=1e-6):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.experts = [expert_class(in_features, out_features) for _ in range(n_experts)]
        self.weights = nn.Parameter(torch.ones(n_experts))

        # epsilon: minimum weight for each expert
        self.eps = eps
        # rho: probability of using any given expert
        self.rho = rho

    def forward(self, x: Tensor):
        rand_i = random.randint(0, len(self.experts) - 1)
        weighted_avg = self.experts[rand_i](x)
        # weighted_avg = torch.zeros((x.shape[0], self.out_features))

        for i, expert in enumerate(self.experts):
            if torch.rand(1) > self.rho:
                continue

            logits = expert(x)
            # weight is rescaled to be > epsilon
            scale = F.relu(self.weights[i]) + self.eps
            weighted_avg += logits * scale

        return weighted_avg


class Example(nn.Module):
    def __init__(self, hidden_width=256, n_experts=1, rho=1.0, eps=1e-6):
        super(Example, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, hidden_width)
        self.wm = WM(hidden_width, 10, nn.Linear, n_experts=n_experts, rho=rho, eps=eps)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 1, 28, 28
        x = self.conv1(x)       # 32, 26, 26
        x = F.leaky_relu(x)
        x = self.conv2(x)       # 64, 24, 24
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)  # 64, 12, 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 9216
        x = self.fc1(x)         # 256
        x = F.leaky_relu(x)
        x = self.wm(x)          # 128
        # x = F.leaky_relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)         # 10
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    total_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_correct += (output.argmax(dim=1) == target).sum().item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    accuracy = total_correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    return accuracy, train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))

    return accuracy, test_loss


# general_config = {
#     'epochs': 10,
#     'lr': 1.0,
#     'gamma': 0.7,
#     'batch_size': 64,
#     'test_batch_size': 1000,
#     'hidden_width': 128,
#     'n_experts': 16,
#     'rho': 0.5,
#     'eps': 1e-2,
# }

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'distribution': 'constant', 'value': 10},
        'lr': {'distribution': 'constant', 'value': 1.0},
        'gamma': {'distribution': 'constant', 'value': 0.7},
        'batch_size': {'distribution': 'constant', 'value': 64},
        'test_batch_size': {'distribution': 'constant', 'value': 1000},

        'hidden_width': {'values': [128, 256, 512]},
        'n_experts': {'values': [1, 4, 8, 16, 32, 64]},
        'rho': {'values': [0.3, 0.5, 0.7, 0.9]},
        'eps': {'values': [1e-6, 1e-4, 1e-2]},
    }
}


def run():
    # wandb.init(project="mnist-wm-sweep", config=general_config)
    wandb.init()

    # Training setup
    torch.manual_seed(8128)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': wandb.config.batch_size}
    test_kwargs = {'batch_size': wandb.config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    # Eval config
    epochs = wandb.config.epochs
    lr = wandb.config.lr
    gamma = wandb.config.gamma

    model = Example(
        hidden_width=wandb.config.hidden_width,
        n_experts=wandb.config.n_experts,
        rho=wandb.config.rho,
        eps=wandb.config.eps,
    ).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train(model, device, train_loader, optimizer, epoch)
        val_acc, val_loss = test(model, device, test_loader)
        scheduler.step()

        wandb.log({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
        })

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


def main():
    # sweep_id = wandb.sweep(sweep=sweep_config, project="mnist-wm-sweep")
    # wandb.agent(sweep_id, function=main, count=2)
    run()


if __name__ == '__main__':
    main()
