# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import glob
import json
import random
from functools import partial
from typing import List, Sequence, Union, Optional, Callable, Any

from .schema import is_valid_schema, SCHEMA, T
from ..misc import parallel_map, ParallelMode



class MMDataset(object):

    SUPPORTED_EXTS = ['.h5', '.json', '.jsonl']

    def __init__(self, items: List[T]=list(), schema: SCHEMA=SCHEMA.MM):
        self._items = items
        self._schema = schema

    @property
    def schema(self) -> SCHEMA:
        return self._schema

    @property
    def items(self) -> List[T]:
        return self._items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index: Union[slice, int]) -> Union['MMDataset', T]:
        if isinstance(index, slice):
            start, stop, stride = index.indices(len(self.items))
            return MMDataset(self._items[start:stop:stride])
        else:
            return self.items[index]

    def __setitem__(self, i, elem):
        self.items[i] = elem

    def __iter__(self):
        self._cur = 0
        return self

    def __next__(self):
        if self._cur < len(self):
            item = self._items[self._cur]
            self._cur += 1
            return item
        else:
            raise StopIteration

    def __add__(self, other: 'MMDataset') -> 'MMDataset':
        return MMDataset(self._items + other._items)

    def __iadd__(self, other: 'MMDataset') -> 'MMDataset':
        self._items.extend(other._items)
        return self

    def extend(self, other: 'MMDataset') -> 'MMDataset':
        self += other
        return self
    
    def append(self, item: T) -> 'MMDataset':
        self._items.append(item)
        return self
    
    def pop(self, index: int=-1) -> T:
        item = self._items.pop(index)
        return item
    
    def sort(self, key: Callable[[T], Any], reverse: bool=False) -> 'MMDataset':
        return MMDataset(sorted(self.items, key=key, reverse=reverse))

    def nonempty(self) -> 'MMDataset':
        return MMDataset([item for item in self._items if item])

    def map(
        self,
        func: Callable[[T], T], 
        *,
        max_workers: int=8, 
        mode: ParallelMode=ParallelMode.THREAD, 
        progress: bool=True, 
        order: bool=True
    ) -> 'MMDataset':
        results = parallel_map(func, self.items, max_workers=max_workers, mode=mode, progress=progress, order=order)
        return MMDataset(results)

    def filter(
        self, 
        func: Callable[[T], bool], 
        *,
        max_workers: int=8, 
        mode: ParallelMode=ParallelMode.THREAD, 
        progress: bool=True, 
        order: bool=True
    ) -> 'MMDataset':
        results = parallel_map(func, self.items, max_workers=max_workers, mode=mode, progress=progress, order=order)
        results = [self.items[i] for i, v in enumerate(results) if v] 
        return MMDataset(results)
    
    def sanitize(
        self, 
        *, 
        schema: Optional[SCHEMA]=None, 
        max_workers: int=1, 
        mode: ParallelMode=ParallelMode.THREAD, 
        progress: bool=True,
        suppress_exceptions: bool=True,
    ) -> 'MMDataset':
        schema = self.schema if schema is None else schema
        func = partial(is_valid_schema, schema=schema, suppress_exceptions=suppress_exceptions)
        return self.filter(func, max_workers=max_workers, mode=mode, progress=progress, order=True)

    def shuffle(self, ) -> 'MMDataset':
        random.shuffle(self._items)
        return self

    def sample(self, k: int) -> 'MMDataset':
        indices = random.sample(range(len(self)), k)
        items = [self.items[i] for i in indices]
        return MMDataset(items)

    @classmethod
    def from_json(cls, path: str, schema: SCHEMA=SCHEMA.MM) -> 'MMDataset':
        with open(path, 'r') as f:
            items = json.load(f)
        return cls(items, schema)

    @classmethod
    def from_jsonl(
        cls, 
        path: str, 
        *,
        schema: SCHEMA=SCHEMA.MM,
        max_workers: int=1,
        mode: ParallelMode=ParallelMode.THREAD,
        progress=False,
    ) -> 'MMDataset':
        with open(path, 'r') as f:
            items = f.read().strip().split('\n')
        items = parallel_map(
            lambda item: json.loads(item),
            items,
            max_workers=max_workers,
            mode=mode,
            order=True,
            progress=progress,
        )
        return cls(items, schema)

    def export_json(
        self, 
        path: str, 
        *,
        indent: int=4, 
        ensure_ascii: bool=False
    ) -> None:
        with open(path, 'w') as f:
            json.dump(self.items, f, ensure_ascii=ensure_ascii, indent=indent)

    def export_jsonl(
        self, 
        path: str, 
        *,
        max_workers: int=1, 
        mode: ParallelMode=ParallelMode.THREAD, 
        progress: bool=False
    ) -> None:
        outputs = parallel_map(
            lambda item: json.dumps(item, separators=(',',':')),
            self.items,
            max_workers=max_workers,
            mode=mode,
            order=True,
            progress=progress,
        )
        with open(path, 'w') as f:
            f.write('\n'.join(outputs))

    # h5 format
    def export_h5(self, output_dir: str, part_name: str, num_h5: int = 32, max_size: int = 100000000, shuffle: bool = True, seed: int = 2023, progress: bool = True, check: bool = False) -> None: ...
    @classmethod
    def from_h5(cls, path: Union[str, List[str]], schema: SCHEMA = ..., *, load_all_at_once: bool = False, max_workers: int = 8, mode: ParallelMode = ..., progress: bool = False) -> 'MMDataset': ...
    def info(self) -> None: ...
    def head(self, n: int) -> None: ...


    @classmethod
    def from_auto(cls, path: Union[str, List[str]], schema: SCHEMA=SCHEMA.MM, **kwargs) -> 'MMDataset':
        return load_mmdataset(path, schema=schema, **kwargs)



def load_mmdataset(path: str, schema: SCHEMA, **kwargs) -> MMDataset:
    '''load mmdataset
    Args:
        path: Union[str, List[str]], 
    '''
    if isinstance(path, str):
        if os.path.isdir(path):
            paths = sorted(glob.glob(os.path.join(path, '*')))
        else:
            paths = sorted(glob.glob(path))
    elif isinstance(path, Sequence):
        paths = path
    else:
        raise AttributeError(f'invalid {path}')

    exts = [os.path.splitext(p)[-1] for p in paths]
    paths = [paths[i] for i, e in enumerate(exts) if e in MMDataset.SUPPORTED_EXTS]
    exts = [e for _, e in enumerate(exts) if e in MMDataset.SUPPORTED_EXTS]
    assert len(paths) > 0, f'invalid {path}'
    
    if len(set(exts)) == 1 and exts[0] == '.h5':
        return MMDataset.from_h5(paths, schema=schema, **kwargs)

    dataset = MMDataset()
    for i, p in enumerate(paths):
        ext = exts[i]
        if ext == '.h5':
            kw = {'load_all_at_once': True}.update(kwargs)
            dataset += MMDataset.from_h5(p, schema, **kw)
        elif ext == '.json':
            dataset += MMDataset.from_json(p, schema, **kwargs)
        elif ext == '.jsonl':
            dataset += MMDataset.from_jsonl(p, schema, **kwargs)

    return dataset
