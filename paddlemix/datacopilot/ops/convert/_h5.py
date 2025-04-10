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
import json
import h5py
import glob
import random
import numpy as np
from pathlib import Path
from contextlib import contextmanager

from typing import List, Union

from ...core import MMDataset, SCHEMA
from ...misc import ParallelMode, parallel_map, freeze_rng_state, enumerate_chunk

__all__ = ['export_h5', 'check_h5', 'from_h5']


@contextmanager
def open_h5(path, max_size):
    of = h5py.File(path, 'w')
    ds = of.create_dataset('dataset', (max_size, ), maxshape=(None, ), dtype='uint8')
    ds_offset = of.create_dataset('offset', (max_size, ), maxshape=(None, ), dtype='uint32')
    offset = [0, 0] # offset, count
    yield of, ds, ds_offset, offset
    of.flush()
    ds.resize((offset[0],)); ds_offset.resize((offset[1],))
    of.close()


def write_h5(dataset, max_size, output_dir, part_name, part, subpart=0):    
    '''write dataset to h5
    '''
    path = os.path.join(
        output_dir,
        '{}-part-{:05d}-{:04d}.h5'.format(part_name, part, subpart)
    )
    with open_h5(path, max_size=max_size) as (of, ds, ds_offset, offset):
        for i, item in enumerate(dataset):
            meta = json.dumps(item, separators=(',',':')).encode()
            meta = np.frombuffer(meta, dtype=np.uint8)
            if len(meta) + offset[0] > max_size:
                write_h5(
                    dataset[i:], 
                    max_size=max_size, 
                    output_dir=output_dir, 
                    part_name=part_name, 
                    part=part, 
                    subpart=subpart+1
                )
                break
            else:
                ds[offset[0]:offset[0]+len(meta)] = meta
                ds_offset[offset[1]] = offset[0]
                offset[0] += len(meta)
                offset[1] += 1
                if i % 1000000 == 0:
                    of.flush()


def export_h5(
    dataset: MMDataset,
    output_dir: str,
    part_name: str,
    num_h5: int=32,
    max_size: int=100*1000*1000,
    shuffle: bool=True,
    seed: int=2023,
    progress: bool=True,
    check: bool=False,
) -> None:
    """export to h5 format
    Args:
        dataset: data to export
        part_name: name of the part, eg. json file name 
        output_dir: output directory
    """
    # TODO, lyuwenyu
    def _preprocess(item):
        return json.dumps(item, separators=(',',':'))

    outputs = dataset.map(_preprocess, max_workers=1, progress=progress)

    if shuffle:
        outputs.shuffle(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _export(params):
        idx, subdata = params
        prefix = os.path.join(output_dir, '{}-part-{:05d}'.format(part_name, idx))

        part = 0
        of = h5py.File(prefix + '_{:04d}.h5'.format(part), 'w')
        ds = of.create_dataset('dataset', (max_size, ), maxshape=(None, ), dtype='uint8')
        ds_offset = of.create_dataset('offset', (max_size, ), maxshape=(None, ), dtype='uint32')
        offset = [0, 0] # offset, count

        for i, item in enumerate(subdata):
            meta = np.frombuffer(item.encode(), dtype=np.uint8)
            
            if len(meta) + offset[0] > max_size:
                of.flush()
                ds.resize((offset[0],)); ds_offset.resize((offset[1],))
                of.close()

                part += 1
                of = h5py.File(prefix + '_{:04d}.h5'.format(part), 'w')
                ds = of.create_dataset('dataset', (max_size, ), maxshape=(None, ), dtype='uint8')
                ds_offset = of.create_dataset('offset', (max_size, ), maxshape=(None, ), dtype='uint32')
                offset = [0, 0]

            ds[offset[0]:offset[0]+len(meta)] = meta
            ds_offset[offset[1]] = offset[0]
            offset[0] += len(meta)
            offset[1] += 1

            if i % 1000000 == 0:
                of.flush()
        
        of.flush()
        ds.resize((offset[0],)); ds_offset.resize((offset[1],))
        of.close()

    parallel_map(
        _export, 
        [(i, sub) for i, sub in enumerate_chunk(outputs, num_chunks=num_h5)],
        max_workers=1,
        mode=ParallelMode.THREAD,
        progress=progress,
        order=False,
    )

    if check:
        with freeze_rng_state(seed):
            st = random.randint(0, len(outputs)-2)
            end = min(st + random.randint(1, 2), len(outputs)-1)
            check_h5(h5_dir=output_dir, start=st, end=end)


def check_h5(h5_dir, start: int=5, end: int=8, verbose: bool=True):
    """load h5 file randomly to check correctness
    """
    dataset = from_h5(h5_dir, load_all_at_once=False)
    if verbose:
        print(f'lenght: {len(dataset)}')

    for i in range(start, end):
        item = dataset[i]
        if verbose:
            print(f'index={i}: {item}')


class _H5Data:
    def __init__(self, h5_files: List[str], ) -> None:
        self.h5_files = h5_files
        self.h5_nums = self._get_nums()
        self.total_num = sum(self.h5_nums)

    def _get_nums(self, ) -> List[int]:
        nums = []
        for _, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                nums.append(f['offset'].shape[0])
        return nums
    
    def tolist(self, ):
        return self[:]
        
    def __len__(self):
        return self.total_num

    def __iter__(self):
        self._cur = 0
        return self

    def __next__(self):
        if self._cur < len(self):
            item = self._getitem(self._cur)
            self._cur += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, stride = i.indices(len(self))
            return [self._getitem(j) for j in range(start, stop, stride)]
        else:
            j = i % len(self) if i < 0 else i
            return self._getitem(j) 

    def _getitem(self, k):
        assert k < len(self), f'index should be less than {len(self)}, but got {k}'
        def _find_position(i):
            cur = 0
            for j, num in enumerate(self.h5_nums):
                if cur + num - 1 < i:
                    cur += num
                else:
                    return j, i - cur

        j, i = _find_position(k)

        with h5py.File(self.h5_files[j], 'r') as f:
            if i == f['offset'].shape[0] - 1:
                st, end = f['offset'][i], f['dataset'].shape[0]
            else:
                st, end = f['offset'][i], f['offset'][i+1]

            meta = f['dataset'][st:end]
            return json.loads(meta.tobytes().decode())


def from_h5(
    path: Union[str, List[str]], 
    schema: SCHEMA=SCHEMA.MM,
    *,
    load_all_at_once: bool=False,
    max_workers: int=8,
    mode: ParallelMode=ParallelMode.THREAD,
    progress: bool=False,
) -> MMDataset:
    """load from h5 file or directory or files
    """    
    if isinstance(path, str):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.h5')))
        else:
            files = sorted(glob.glob(path))
        assert len(files) > 0, f'invalid {path}'
    elif isinstance(path, (tuple, list)):
        files = path
    else:
        raise AttributeError(f'not support {path}')

    for f in files:
        assert os.path.exists(f) and os.path.splitext(f)[-1] == '.h5', \
            f'{f} is not valid h5 file'
        
    if load_all_at_once:
        data = _H5Data(files)
        items = parallel_map(
            lambda i: data[i], 
            range(len(data)), 
            max_workers=max_workers,
            mode=mode,
            progress=progress
        )
    else:
        items = _H5Data(files)

    return MMDataset(items, schema)


MMDataset.export_h5 = export_h5 # type: ignore
MMDataset.from_h5 = staticmethod(from_h5) # type: ignore
