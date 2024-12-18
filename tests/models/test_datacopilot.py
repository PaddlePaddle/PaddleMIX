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
import pytest

from paddlemix.datacopilot.core import MMDataset, SCHEMA, is_valid_schema
from paddlemix.datacopilot.misc import open_tmp_dir


@pytest.fixture
def dataset():
    blob0 = \
    {
        'id': '0',
        'image': 'http://0.jpg',
        'conversations': 
        [
            {
                'from': 'human', 
                'value': 'aaa'
            }, 
            {
                'from': 'gpt', 
                'value': 'aaaaa'
            }
        ]
    }
    blob1 = \
    {
        'id': '1',
        'image': 'http://1.jpg',
        'conversations': 
        [
            {
                'from': 'human', 
                'value': 'xxxx'
            }, 
            {
                'from': 'gpt', 
                'value': 'xxx'
            }
        ]
    }
    blob2 = \
    {
        'id': '2',
        'image': 'http://2.jpg',
        'conversations': 
        [
            {
                'from': 'human', 
                'value': 'yyy'
            }, 
            {
                'from': 'gpt', 
                'value': 'yyyy'
            }
        ]
    }
    blob3 = \
    {
        'id': '3',
        'image': 'http://3.jpg',
        'conversations': 
        [
            {
                'from': 'human', 
                'value': 'zzz'
            }, 
            {
                'from': 'gpt', 
                'value': 'zzzz'
            }
        ]
    }
    dataset = MMDataset([blob0, blob1, blob2, blob3], schema=SCHEMA.MM)
    return dataset


def test_schema(dataset: MMDataset):
    assert len(dataset) == 4, ''
    assert is_valid_schema(dataset[0], SCHEMA.MM) == True, ''


def test_dataset(dataset: MMDataset):
    assert len(dataset[1:2]) == 1, ''
    assert len(dataset[-2:-1]) == 1, ''
    assert dataset[0] == dataset[-len(dataset)], ''
    
    for data in dataset:
        assert isinstance(data, dict), ''

    for data in dataset:
        assert isinstance(data, dict), ''



def test_h5(dataset: MMDataset):

    with open_tmp_dir() as H5ROOT:

        NUM = 3
        NAME = 'test'

        dataset = dataset[:3] + dataset.shuffle()
        assert len(dataset) == 3 + 4, ''

        dataset.export_h5(
            H5ROOT, 
            part_name=NAME, 
            shuffle=False, 
            num_h5=NUM,
            progress=False,
            check=True,
        )

        import os
        import glob 

        files = sorted(glob.glob(os.path.join(H5ROOT, '*.h5')))
        newdataset = dataset.from_h5(files)

        for i in range(len(dataset)):
            assert dataset[i] == newdataset[i], ''


def test_op(dataset: MMDataset):

    def update_image_path(item):
        if item:
            item['image'] = os.path.join('/root', item['image'])
            return item
        else:
            return None
            
    newdataset = dataset.map(update_image_path, ).nonempty()
    assert len(newdataset) == len(dataset), ''



if __name__ == '__main__':
    pytest.main([__file__])