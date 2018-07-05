from __future__ import absolute_import, print_function

import numpy as np
import os
import nibabel
import tensorlayer as tl
from util.data_loader import *

class DataAug():
    def __init__(self, config):
        self.config    = config
        self.data_root = config['data_root'] if type(config['data_root']) is list else [config['data_root']]
        self.modality_postfix     = config.get('modality_postfix', ['flair','t1', 't1ce', 't2'])
        self.with_ground_truth    = config.get('with_ground_truth', False)
        self.label_postfix = config.get('label_postfix', 'seg')
        self.file_postfix  = config.get('file_postfix', 'nii.gz')
        self.data_names    = config.get('data_names', None)
        self.data_num      = config.get('data_num', None)
        self.aug_mode     = config.get('aug_mode', False)
        self.aug_save_path     = config.get('aug_save_path')
        self.aug_modes     = config.get('aug_modes', ['flip1_mode', 'flip2_mode', 'rotation_mode', 'shift_mode', 'shear_mode', 'zoom_mode', 'elastic_mode'])

    def distort_imgs(self, data, mode):
		""" data augumentation """
		x1, x2, x3, x4, y = data
		if(mode == 'flip1_mode'):
			x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis=0, is_random=True) # up down
		if(mode == 'flip2_mode'):
			x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis=1, is_random=False) # left right
		if(mode == 'elastic_mode'):
			x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y], alpha=720, sigma=24, is_random=True)
		if(mode == 'rotation_mode'):
			x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20, is_random=True, fill_mode='constant') # nearest, constant
		if(mode == 'shift_mode'):
			x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
		if(mode == 'shear_mode'):
			x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05, is_random=True, fill_mode='constant')
		if(mode == 'zoom_mode'):
			x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y], zoom_range=[0.9, 1.1], is_random=True, fill_mode='constant')
		return x1, x2, x3, x4, y

    def vis_imgs(self, X, y, path):
        """ show one slice """
        if y.ndim == 2:
            y = y[:,:,np.newaxis]
        assert X.ndim == 3
        tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
            X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
            X[:,:,3,np.newaxis], y]), size=(1, 5),
            image_path=path)

    def vis_imgs2(self, X, y_, y, path):
        """ show one slice with target """
        if y.ndim == 2:
            y = y[:,:,np.newaxis]
        if y_.ndim == 2:
            y_ = y_[:,:,np.newaxis]
        assert X.ndim == 3
        tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
            X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
            X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
            image_path=path)

    def aug_data(self):
        dataloader = DataLoader(self.config)
        dataloader.load_data()

        image_num = dataloader.get_total_image_number()
        #modes = ['flip1', 'flip2', 'elastic', 'rotation', 'shift', 'shear', 'zoom']

        for i in range(image_num):
            if(self.aug_mode):
            	for mode in self.aug_modes:
	                [volume, label, temp_weight, temp_name, temp_bbox, temp_size] = dataloader.get_image_data_label_with_name(i)
	                volume = np.asarray(volume)
	                x_flair_test = volume[0,70,:,:,np.newaxis]
	                x_t1_test = volume[1,70,:,:,np.newaxis]
	                x_t1ce_test= volume[2,70,:,:,np.newaxis]
	                x_t2_test = volume[3,70,:,:,np.newaxis]
	                y = label[70,:,:,np.newaxis]
	                X_r = np.concatenate((x_flair_test, x_t1_test, x_t1ce_test, x_t2_test), axis=2)
	                
	                fullpath = self.aug_save_path+"/{0:}_aug".format(temp_name)+'_'+mode
	                #basedir = os.path.dirname(fullpath)
	                if not os.path.exists(fullpath):
	                    os.makedirs(fullpath)

	                x_flair_r,x_t1_r,x_t1ce_r,x_t2_r,la = self.distort_imgs([x_flair_test,x_t1_test,x_t1ce_test,x_t2_test,y], mode)
	                x_dis = np.concatenate((x_flair_r, x_t1_r, x_t1ce_r, x_t2_r), axis=2)

	                patient_frames_flair = []
	                patient_frames_t1 = []
	                patient_frames_t1ce = []
	                patient_frames_t2 = []
	                patient_frames_l = []
	                for frame in range(155):
	                    x_flair, x_t1, x_t1ce, x_t2, l = self.distort_imgs([volume[0,frame,:,:,np.newaxis], volume[1,frame,:,:,np.newaxis], volume[2,frame,:,:,np.newaxis], volume[3,frame,:,:,np.newaxis], label[frame,:,:,np.newaxis]], mode)
	                    patient_frames_flair.append(x_flair)
	                    patient_frames_t1.append(x_t1)
	                    patient_frames_t1ce.append(x_t1ce)
	                    patient_frames_t2.append(x_t2)
	                    patient_frames_l.append(l)

	                #result_flair = np.dstack(patient_frames_flair)
	                #result_t1 = np.dstack(patient_frames_t1)
	                #result = nibabel.Nifti1Image(result_t1, affine=np.eye(4))
	                #nibabel.save(result, fullpath)

	                mode = '_' + mode + '.'


	                result_flair = np.dstack(patient_frames_flair)
	                result = nibabel.Nifti1Image(result_flair, affine=np.eye(4))
	                datapath = fullpath+"/{0:}_flair_aug".format(temp_name)+mode+self.file_postfix
	                nibabel.save(result, datapath)

	                result_t1 = np.dstack(patient_frames_t1)
	                result = nibabel.Nifti1Image(result_t1, affine=np.eye(4))
	                datapath = fullpath+"/{0:}_t1_aug".format(temp_name)+mode+self.file_postfix
	                nibabel.save(result, datapath)

	                result_t1ce = np.dstack(patient_frames_t1ce)
	                result = nibabel.Nifti1Image(result_t1ce, affine=np.eye(4))
	                datapath = fullpath+"/{0:}_t1ce_aug".format(temp_name)+mode+self.file_postfix
	                nibabel.save(result, datapath)

	                result_t2 = np.dstack(patient_frames_t2)
	                result = nibabel.Nifti1Image(result_t2, affine=np.eye(4))
	                datapath = fullpath+"/{0:}_t2_aug".format(temp_name)+mode+self.file_postfix
	                nibabel.save(result, datapath)

	                result_l = np.dstack(patient_frames_l)
	                result = nibabel.Nifti1Image(result_l, affine=np.eye(4))
	                datapath = fullpath+"/{0:}_".format(temp_name)+self.label_postfix+"_aug"+mode+self.file_postfix
	                nibabel.save(result, datapath)

	                self.vis_imgs(X_r,y, fullpath+"/{0:}_before.png".format(temp_name))
	                self.vis_imgs(x_dis,la, fullpath+"/{0:}_after.png".format(temp_name))