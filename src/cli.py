import inspect
import sys
from pathlib import Path

from utils import _

def pattern_basename(x):
  if x.suffixes and x.suffixes[-1] in ['.xcli', '.cli']:
    return x.with_name(x.stem + ''.join(x.suffixes[:-1]))
  return x.with_name(x.name)

def _pattern_files_bn(bn):
  yield bn.with_name(bn.name + '.xcli'), True
  yield bn.with_name(bn.name + '.cli'), False

class Pattern:
  def __init__(self, name, p, data=None):
    self.name = name
    self.path = p

    if data is None:
      with p.open('r') as f:
        self.data = f.read()
    else:
      self.data = data

  def __str__(self):
    return f'CLI Pattern {repr(self.name)} @ {repr(self.path)}'

  def __repr__(self):
    return f'Pattern({repr(self.name)}, {repr(self.path)})'

  def __call__(self, *args, **kwargs):
    for l in self.data:
      res = ''
      escape = False
      subs = False

      i = -1
      length = len(l)
      while i < length - 1:
        i += 1
        x = l[i]

        if escape:
          res += x
          escape = False
          continue

        if subs:
          expr = ''
          while x != '}':
            expr += x
            i += 1
            x = l[i]
          res += str(kwargs[expr])
          subs = False
          continue

        if x == '\\':
          escape = True
          continue
        if x == '{':
          subs = True
          continue

        res += x

      sys.stdout.write(res)

def read_xtended(p):
  cur_key = None
  data = []

  res = _()
  def commit_pat():
    nonlocal cur_key, data
    if cur_key is None:
      return

    cur_key = cur_key.replace(' ', '_')

    pat = Pattern(cur_key, p, data)
    if cur_key in res:
      raise ValueError(f'duplicate pattern {repr(cur_key)}: found {repr(pat)} and {repr(res[cur_key])}')

    res[cur_key] = pat
    cur_key = None
    data = []

  with p.open('r') as f:
    for l_raw in f.readlines():
      l = l_raw.strip()

      if cur_key is None:
        if not l:
          continue

        cur_key = l
        continue

      if l == '\\end':
        commit_pat()
        continue

      data.append(l_raw)

  commit_pat()

  return res

class _Patterns:
  def __init__(self, root=None):
    self._cache = _()
    self._root = root

  def __getattr__(self, k):
    assert self._root is not None

    if k not in self._cache:
      res = PatternHierarchy(k, root=self._root/k)
      res._enumerate_patterns()
      if not res._cache:
        # todo(maximsmol): log which files were considered
        raise ValueError(f'found no patterns for {repr(k)}')
      self._cache[k] = res

    return self._cache[k]

class PatternHierarchy(_Patterns):
  def __init__(self, key, root=None):
    super().__init__(root)
    self._key = key

  def __call__(self, *args, **kwargs):
    if '_root' not in self._cache:
      raise ValueError(f'pattern hierarchy {repr(self._key)} has no root definition')

    self._cache._root(*args, **kwargs)

  def _enumerate_patterns(self):
    def add_pattern(k, pat):
      if k in self._cache:
        real_key = self.key if k == '_root' else k
        raise ValueError(
          f'duplicate pattern definition for {repr(real_key)}: '
          f'found {repr(pat)} and {repr(self._cache[k])}')
      self._cache[k] = pat

    if self._root.is_dir():
      for p1 in self._root.iterdir():
        k1 = pattern_basename(p1).name
        add_pattern(k1, PatternHierarchy(k1, root=p1))

    for p1, xtended in _pattern_files_bn(self._root):
      if not p1.exists():
        continue

      if xtended:
        for k, v in read_xtended(p1).items():
          add_pattern(k, v)
        continue

      add_pattern('_root', Pattern(k, p1))

patterns = _Patterns()
def use_default_root():
  caller = inspect.stack()[1]
  module = inspect.getmodule(caller[0])
  filename = module.__file__

  patterns._root = Path(filename).parent / 'cli_patterns'
