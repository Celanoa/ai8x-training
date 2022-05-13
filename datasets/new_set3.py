###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
CIFAR-100 Dataset
"""
from genericpath import exists
import os

import torchvision
import torch
from torchvision import datasets, transforms

import ai8x



def test_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data

    if load_train:
        train_file = os.path.join(data_dir, 'RoCoLe') + '/train.pt'
        train_dataset = torch.load(train_file)
                                
    else:
        train_dataset = None

    if load_test:
        test_file = os.path.join(data_dir, 'RoCoLe') + '/test.pt'
        test_dataset = torch.load(test_file)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'RoCoLeNew3',
        'input': (3, 64, 64),
        'output': ('Healthy', 'Unhealthy'),
        'loader': test_get_datasets,
    },
]
