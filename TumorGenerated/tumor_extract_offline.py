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
import nibabel as nib

TUMOR_LABEL = 2
ORGAN_LABEL = 1

def get_bbox_from_mask(mask, outside_value=0, pad=0):
    t = time.time()
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


class TumorExtractor(object):
    def __init__(self, image_root, label_root, num_processes=6):
                
        self.num_processes = num_processes
        self.image_root, self.label_root= image_root, label_root,
        self.patient_identifiers = self.get_data_identifiers( image_root, label_root)
        self.savepath = '/home/plyu/Documents/projects/tumors/liver_tumor_dataset/liver_tumor_seg/TumorGenerated/off_tumor'
   
    def extract(self, case):
        image = nib.load(os.path.join(self.image_root,case[0]))
        data = image.get_fdata()
        # print(0)
        mask = nib.load(os.path.join(self.label_root,case[1]))
        seg = mask.get_fdata()
        # print(np.unique(seg))

        # assert len(case)==2, "only supports single modality"
        # data = case[0]
        # seg = case[1]
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

        out_name = case[1].split('.nii.gz')
        out_name[0] += '_tumor'
        out_name[1] = 'nii.gz'
        out_name = '.'.join(out_name)
        out_name = os.path.join(self.savepath,out_name)
        print(out_name)
        # nib.Nifti1Image(np.array(tumors),mask.affine).to_filename(out_name)
        # np.save(out_name, np.array(tumors))

    def extract_all(self):
        p = Pool(self.num_processes)
        res = p.map(self.extract, self.patient_identifiers)
        p.close()
        p.join()

    def get_data_identifiers(self, image_root, label_root):
        img = [x for x in np.sort(os.listdir(image_root)) if x.endswith('.nii.gz')]
        # print(len(img))
        label = [x for x in np.sort(os.listdir(label_root)) if x.endswith('.nii.gz')]
        # print(len(label))

        return zip(img,label)

if __name__ == "__main__":

    # data_root = '../data/nnUNet_preprocessed/Task040_KiTS/nnUNetData_plans_v2.1_stage1/'
    image_root = '/home/plyu/Documents/projects/tumors/liver_tumor_dataset/04_LiTS/img/'
    label_root = '/home/plyu/Documents/projects/tumors/liver_tumor_dataset/04_LiTS/label/'
    tumor_extractor = TumorExtractor(image_root, label_root, num_processes=1)
    tumor_extractor.extract_all()

