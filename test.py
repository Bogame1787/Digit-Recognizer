from custom import VisionTransformer
import torch
import torchvision.transforms as tt
from PIL import Image
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


def predict(path, model, device):

    img = Image.open(path).convert("L")

    tfms = tt.Compose(
        [
        tt.ToTensor(),
        tt.Resize((32,32))
        ]
    )

    img = tfms(img).to(device)

    img = 1 - img
    print(img.shape)

    plt.imshow(img.squeeze(0).cpu().detach())
    plt.show()
    img = img.unsqueeze(0)
    output = model(img)
    print(MNIST.classes[torch.argmax(output)])