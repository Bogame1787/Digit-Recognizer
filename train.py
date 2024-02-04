import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from custom import VisionTransformer
from resnet import ResNet9

from torchsummary import summary

np.random.seed(0)
torch.manual_seed(0)

def fit(model, N_EPOCHS, LR, train_loader, test_loader, device):
    
    print("Using device: ", device,
          f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            # print(x.shape)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1)
                                 == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

def main():
    transform = ToTensor()

    train_set = MNIST(root='./data', train=True,
                      download=True, transform=transform)
    test_set = MNIST(root='./data', train=False,
                     download=True, transform=transform)  

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = VisionTransformer(img_size=28, patch_size=4, in_chans=1, n_classes=10, embed_dim=768, depth=6, n_heads=6, mlp_ratio=4, qkv_bias=True, p=0.1, attn_p=0.1).to(device)
    model = ResNet9(1, 10).to(device)

   
    fit(model, 15, 0.01, train_loader, test_loader,device)
    fit(model, 10, 0.001, train_loader, test_loader, device)
    fit(model, 10, 0.0001, train_loader, test_loader, device)


    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
