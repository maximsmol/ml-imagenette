from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LocalResponseNorm, ReLU, Flatten, Linear
from torch.nn import Sigmoid, LeakyReLU, PReLU, ReLU6, RReLU, CELU, GELU, Tanh, Softplus, SiLU

from mish import Mish

class AlexNetLike(Module):
  def __init__(self):
    super().__init__()

    act = lambda: Mish()

    self.model = Sequential(
      Conv2d(in_channels=3, out_channels=96,
             kernel_size=(11, 11), stride=(4, 4), padding=2),
      LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
      MaxPool2d(kernel_size=(3, 3), stride=2),
      act(),

      Conv2d(in_channels=96, out_channels=256,
             kernel_size=(5, 5), padding=2),
      LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
      MaxPool2d(kernel_size=(3, 3), stride=2),
      act(),

      Conv2d(in_channels=256, out_channels=384,
             kernel_size=(3, 3), padding=1),
      act(),

      Conv2d(in_channels=384, out_channels=384,
             kernel_size=(3, 3), padding=1),
      act(),

      Conv2d(in_channels=384, out_channels=256,
             kernel_size=(3, 3), padding=1),
      LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
      MaxPool2d(kernel_size=(3, 3), stride=2),
      act(),

      Flatten(),
      Linear(in_features=9216, out_features=4096),
      act(),
      Linear(in_features=4096, out_features=4096),
      act(),
      Linear(in_features=4096, out_features=10))

  def forward(self, x):
    return self.model(x)
