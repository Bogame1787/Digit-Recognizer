import torch
from torchvision.models import resnet50
from torchvision.datasets import MNIST as MNIST_data
from torchvision import transforms

batch_size = 64

train_dataset = MNIST_data('/kaggle/input/digit-recognizer/train.csv', transform= transforms.Compose(
                            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data('/kaggle/input/digit-recognizer/test.csv')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, shuffle=False)

model = resnet50(pretrained = True)

