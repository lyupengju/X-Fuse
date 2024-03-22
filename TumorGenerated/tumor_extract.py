import numpy as np
import SimpleITK as sitk
import os
import time
from multiprocessing import Pool
from skimage.morphology import label
import random
from scipy import ndimage
import argparse
from skimage.transform import resize
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter

TUMOR_LABEL = 2
ORGAN_LABEL = 1

def get_bbox_from_mask(mask, outside_value=0, pad=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0])) - pad
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + pad
    minxidx = int(np.min(mask_voxel_coords[1])) - pad
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + pad
    minyidx = int(np.min(mask_voxel_coords[2])) - pad
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + pad

    if (maxzidx - minzidx)%2 != 0:
        maxzidx += 1
    if (maxxidx - minxidx)%2 != 0:
        maxxidx += 1
    if (maxyidx - minyidx)%2 != 0:
        maxyidx += 1
    return np.array([[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]], dtype=np.int)

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]+1), slice(bbox[1][0], bbox[1][1]+1), slice(bbox[2][0], bbox[2][1]+1))
    return image[resizer]

def get_slicer_from_bbox(bbox):
    slicer = []
    for (x,y) in bbox:
        slicer.append([(x-y)// 2, (y-x) // 2 + 1])
    return np.array(slicer, dtype=np.int)

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), 
                            sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices

def rotate_coords_3d(coords, angle):
    rot_matrix = np.identity(len(coords))
    rotation = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    rot_matrix = np.dot(rot_matrix, rotation)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords

def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)

'''
def do_rotation(data, seg, degree=180):
    # print('rotation')
    # print(f'rotation: seg_volume before: {np.sum(seg>0)}',end=' ')
    data = np.pad(data, 20, 'constant', constant_values=0)
    seg = np.pad(seg, 20, 'constant', constant_values=0)
    degree = 2*(np.random.rand()-0.5)*degree
    data = ndimage.rotate(data, degree, reshape=False, axes=(1, 2), order=3)
    seg = ndimage.rotate(seg, degree, reshape=False, axes=(1, 2), order=0)
    # print(f'seg_volume after: {np.sum(seg>0)}')
    # print(np.unique(seg))
    return data, seg

def do_elastic(data, seg, alpha=(0.,900.),sigma=(9.,13.)):
    # print('elastic')
    # print(f'elastic: seg_volume before: {np.sum(seg>0)}',end=' ')
    start_time = time.time()
    
    coords = create_zero_centered_coordinate_mesh(data.shape)
    a = np.random.uniform(alpha[0], alpha[1])
    s = np.random.uniform(sigma[0], sigma[1])
    coords = elastic_deform_coordinates(coords, a, s)
    
    time_checkpoint1 = time.time()
    # print('time_checkpoint1', time_checkpoint1 - start_time)
    
    for d in range(len(data.shape)):
        coords[d] += int(np.round(data.shape[d]/2.))
        
    time_checkpoint2 = time.time()
    # print('time_checkpoint2', time_checkpoint2 - time_checkpoint1)
    
    data = interpolate_img(data, coords, order=3, mode='constant',cval=0)
    seg = interpolate_img(seg, coords, order=0, mode='constant',cval=0, is_seg=True)
    # print(f'seg_volume after: {np.sum(seg>0)}')
    # print(np.unique(seg))
    time_checkpoint3 = time.time()
    # print('time_checkpoint3', time_checkpoint3 - time_checkpoint2)
    
    return data, seg

def do_scaling(data, seg, scale=(0.75, 1.25)):
    # print('scaling')
    s = np.random.uniform(scale[0], scale[1])
    new_shape = []
    for x in data.shape:
        new_x = int(x*s)
        if new_x%2 != 0:
            new_x += 1
        new_shape.append(new_x)
    data = resize(data, new_shape, order=3,
                  mode='edge', anti_aliasing=False, cval=0).astype(data.dtype)
    # print(np.unique(seg))
    seg = resize(seg, new_shape, order=0,
                mode="constant",cval=0,
                preserve_range=True, anti_aliasing=False, ).astype(seg.dtype)
    # print(np.unique(seg))
    # print(f'seg_volume after: {np.sum(seg>0)}')
    return data, seg

def do_scaling(data, seg, scale=(0.75, 1.25)):
    # print('scaling')
    start_time = time.time()
    
    coords = create_zero_centered_coordinate_mesh(data.shape)
    sc = np.random.uniform(scale[0], scale[1])
    coords = coords * sc
    new_shape = []
    
    time_checkpoint1 = time.time()
    # print('time_checkpoint1', time_checkpoint1 - start_time)
    
    for d in range(len(data.shape)):
        coords[d] += int(np.round(data.shape[d]/2.))
        
    time_checkpoint2 = time.time()
    # print('time_checkpoint2', time_checkpoint2 - time_checkpoint1)
    
    data = interpolate_img(data, coords, order=3, mode='constant',cval=0)
    seg = interpolate_img(seg, coords, order=0, mode='constant',cval=0, is_seg=True)
    
    time_checkpoint3 = time.time()
    # print('time_checkpoint3', time_checkpoint3 - time_checkpoint2)
    return data, seg
'''

def do_spatial_augment(data, seg, 
                       do_scaling=False, p_scale=0.5, scale=(0.75, 1.25),
                       do_rotation=False, p_rot=0.5, angle=(0, 2 * np.pi),
                       do_elastic=False, p_elas=0.5, alpha=(0.,1000.),sigma=(10., 13.)):
                       
    coords = create_zero_centered_coordinate_mesh(data.shape)
    modified_coords = False
    if do_elastic and np.random.uniform() <= p_elas:
        a = np.random.uniform(alpha[0], alpha[1])
        s = np.random.uniform(sigma[0], sigma[1])
        coords = elastic_deform_coordinates(coords, a, s)
        modified_coords = True
    if do_rotation and np.random.uniform() <= p_rot:
        a = np.random.uniform(angle[0], angle[1])
        coords = rotate_coords_3d(coords, a)
        modified_coords = True
    if do_scaling and np.random.uniform() <= p_scale:
        sc = np.random.uniform(scale[0], scale[1])
        coords = coords * sc
        modified_coords = True

    if modified_coords:
        for d in range(len(data.shape)):
            coords[d] += int(np.round(data.shape[d]/2.))
        data = interpolate_img(data, coords, order=3, mode='constant',cval=0)
        seg = interpolate_img(seg, coords, order=0, mode='constant',cval=0, is_seg=True)
    
    return data, seg

def do_flipping(data, seg):
    # print('flipping')
    # x, y, z: 0, 1, 2
    # print(f'flipping: seg_volume before: {np.sum(seg>0)}',end=' ')
    m = np.random.randint(0,8)
    axis = [None, (2,), (1,), (2,1), 
            (0,), (2,0), (1,0), (2,1,0)][m]
    if axis is not None:
        data = np.flip(data, axis)
        seg = np.flip(seg, axis)
    # print(f'seg_volume after: {np.sum(seg>0)}')
    # print(np.unique(seg))
    return data, seg

def do_gamma(data, seg, retain_stats=True, gamma_range=(0.7, 1.5),epsilon=1e-7):
    # print('gamma')
    # print(f'gamma: seg_volume before: {np.sum(seg>0)}',end=' ')
    if retain_stats:
        mn = np.mean(data)
        sd = np.std(data)
    if np.random.random() < 0.5 and gamma_range[0] < 1:
        gamma = np.random.uniform(gamma_range[0], 1)
    else:
        gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
    minm = np.min(data)
    rnge = np.max(data) - minm
    data = np.power(((data - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
    if retain_stats:
        data = data - np.mean(data) + mn
        data = data / (np.std(data) + 1e-8) * sd
    # print('gamma done')
    # print(f'seg_volume after: {np.sum(seg>0)}')
    # print(np.unique(seg))
    return data, seg

def do_blurring(data, seg, blur_sigmma=(0.5, 1.)):
    # print('blurring')
    sigma = random.uniform(blur_sigmma[0], blur_sigmma[1])
    data = gaussian_filter(data, sigma, order=0)
    return data, seg 

def extract_tumor(mask, center, slicer, data, seg):
    cropped_data = mask\
                * data[center[0]+slicer[0,0]: center[0]+slicer[0,1], 
                        center[1]+slicer[1,0]: center[1]+slicer[1,1],
                        center[2]+slicer[2,0]: center[2]+slicer[2,1]]

    cropped_seg = mask\
                * seg[center[0]+slicer[0,0]: center[0]+slicer[0,1], 
                        center[1]+slicer[1,0]: center[1]+slicer[1,1],
                        center[2]+slicer[2,0]: center[2]+slicer[2,1]]
    return cropped_data, cropped_seg

def get_valid_center(center, patch_length, volume_length):
    patch_offset = [0, patch_length]
    # print(patch_offset,volume_length)
    if (center - patch_length//2) > 0:
        # print(1)
        start = (center - patch_length//2)
    else:
        start = 0
        patch_offset[0] = patch_length//2 - center
        # print(2)

    if (center + patch_length - patch_length//2) <= volume_length:
        end = (center + patch_length - patch_length//2)
        # print(3)
    else:
        end = volume_length
        patch_offset[1] = -(center + patch_length - patch_length//2 - volume_length)
        # print(4)
    # start = (center - patch_length//2) if ((center - patch_length//2) > 0) else 0
    # end = (center + patch_length//2) if ((center + patch_length//2) < volume_length) else volume_length
    # print(start, end, patch_offset)
    return start, end, patch_offset






class TumorExtractor(object):
    def __init__(self, data_root, num_processes=6):
                
        self.num_processes = num_processes
        self.data_folder = data_root
        self.patient_identifiers = self.get_data_identifiers(self.data_folder)
        
    def extract(self, img, mask):
       
        data = img
        seg = mask
        labelmap, numregions = label(seg == TUMOR_LABEL, return_num=True)
        if numregions == 0:
            return
        tumors = []
        for l in range(1, numregions + 1):
            bbox = get_bbox_from_mask(labelmap==l, pad = 0)
            cropped_data = crop_to_bbox(data, bbox)
            cropped_seg = crop_to_bbox(labelmap==l,bbox).astype(np.uint8)
            cropped_seg[cropped_seg>0] = TUMOR_LABEL
            tumors.append([cropped_data, cropped_seg])

        #### organ position ####
        # we need to find out where the classes are and sample some random locations
        # let's do 10,000 samples per class
        # seed this for reproducibility!
        '''
        num_samples = 10000
        min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(12345)
        all_organ_locs = np.argwhere(seg == ORGAN_LABEL)
        target_num_samples = min(num_samples, len(all_organ_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_organ_locs) * min_percent_coverage)))
        selected = all_organ_locs[rndst.choice(len(all_organ_locs), target_num_samples, replace=False)]

        tumors.append(selected)
        '''

if __name__ == "__main__":

    data_root = '../data/nnUNet_preprocessed/Task040_KiTS/nnUNetData_plans_v2.1_stage1/'
    tumor_extractor = TumorExtractor(data_root, num_processes=1)
    tumor_extractor.extract_all()


