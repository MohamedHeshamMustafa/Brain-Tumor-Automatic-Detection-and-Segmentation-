# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function
import numpy as np
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from util.H_data_loader import *
from util.H_data_process import *
from util.parse_config import parse_config
# python H_test.py config17/H_config.txt
def test(config_file):
	# 1, load configure file
	config = parse_config(config_file)
	config_data = config['data']
	print ('Hello world')
	print (config_data['data_root'])

	# 4, load test images
	dataloader = DataLoader(config_data)
	dataloader.load_data()
	image_num = dataloader.get_total_image_number()
	print('Number of loaded patients:' )
	print (image_num)
	#[temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
      
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python test.py config17/test_all_class.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    test(config_file)
    
    
