import torch

def summarize(x):
  if isinstance(x, torch.utils.data.Dataset):
    print(f'Dataset {x.__class__.__name__}')
    print(f'  Length: {len(x)}')
    if hasattr(x, 'root') and x.root is not None:
      print(f'  Root: {x.root}')

class Namespace(dict):
  def __getattr__(self, k):
    if k in self:
      return self[k]
    raise AttributeError(f'{repr(self.__class__.__name__)} object has no attribute {repr(k)}')

  def __setattr__(self, k, v):
    self[k] = v

_ = Namespace
