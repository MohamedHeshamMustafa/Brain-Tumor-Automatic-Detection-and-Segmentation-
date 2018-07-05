from __future__ import absolute_import, print_function
import numpy as np
import sys
import tensorflow as tf
from util.data_loader import *
from util.data_aug import *
from util.parse_config import parse_config

def test(config_file):
	# 1, load configure file
	config = parse_config(config_file)
	config_data = config['data']

	# 2, Augmentation 
	dataaug = DataAug(config_data)
	dataaug.aug_data()
   
if __name__ == '__main__':
 if(len(sys.argv) != 2):
     print('Number of arguments should be 2. e.g.')
     print('    python test.py config17/test_all_class.txt')
     exit()
 config_file = str(sys.argv[1])
 assert(os.path.isfile(config_file))
 test(config_file)