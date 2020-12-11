from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch_optimizer import RAdam, AdaBelief, Yogi, Ranger
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomCrop, Normalize, ToTensor
import torchsummary

import wandb

import repro
import utils

from models.AlexNetLike import AlexNetLike

wandb.init(project='imagenette')

class RandomCropIfNeeded:
  def __init__(self, size):
    self.size = size
    self.cropper = RandomCrop(size)

  def __call__(self, x):
    w, h = x.size

    if w >= self.size:
      hp = 0
    else:
      hp = int((self.size - w) / 2)

    if h >= self.size:
      vp = 0
    else:
      vp = int((self.size - h) / 2)

    padding = (hp + (1 if (self.size - w) % 2 == 0 else 0),
               vp + (1 if (self.size - h) % 2 == 0 else 0),
               hp, vp)

    res = tvF.pad(x, padding, 0, 'constant')
    res = tvF.crop(res,
                   randint(0, max(0, w - self.size)),
                   randint(0, max(0, h - self.size)),
                   self.size, self.size)

    return res

# todo(maximsmol): torch-script this
trans = Compose([
    RandomCropIfNeeded(224),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset_root = Path(__file__).parent.parent / 'imagenette2'
dataset = ImageFolder(dataset_root / 'train', transform=trans)

model = AlexNetLike()
wandb.watch(model, log_freq=3)
# print(torchsummary.summary(model, input_size=(3, 224, 224), device='cpu'))


if torch.cuda.is_available():
  print('Using CUDA')
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

model = model.to(device)
dl = DataLoader(dataset,
                batch_size=128, shuffle=True,
                num_workers=12, pin_memory=True,
                persistent_workers=False)
# dl = DataLoader(dataset,
#                 batch_size=128, shuffle=True)


# opt = SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
# opt = AdamW(model.parameters(), weight_decay=0.01, amsgrad=False)
# opt = Adam(model.parameters())
# opt = AdaBelief(model.parameters())
# opt = Yogi(model.parameters(), weight_decay=5e-4)
opt = Ranger(model.parameters())
# opt = Ranger(model.parameters(), lr=0.02)

# opt = RAdam(model.parameters(), weight_decay=5e-4)
# opt = RAdam(model.parameters())

# sched = StepLR(opt, 250, gamma=0.5)
sched = CosineAnnealingWarmRestarts(opt, 750)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

model.train()
global_step = 0
epoch = 0
while True:
# for j in range(50):
  for batch, (inp, target) in enumerate(dl):
    inp, target = inp.to(device), target.to(device)
    opt.zero_grad()

    out = model(inp)

    loss = F.cross_entropy(out, target)
    loss.backward()

    opt.step()
    sched.step()

    acc = accuracy(F.softmax(out), target)[0].item()
    loss_i = loss.item()

    print(f'{epoch:2}/{batch:3} Loss: {loss_i:.7f} Accuracy: {acc:.7f} LR: {sched.get_last_lr()[0]:.7f}')
    wandb.log({
      'loss': loss_i,
      'accuracy': acc,
      'lr': sched.get_last_lr()[0]
      }, step=global_step)

    global_step += 1
  epoch += 1
