### Tumor Generateion
import random
import cv2
import elasticdeform
import numpy as np
from skimage.morphology import label
import torch
from scipy.ndimage import gaussian_filter,map_coordinates
TUMOR_LABEL = 2
ORGAN_LABEL = 1
cp_configs = {
    "do_cp": True,
    "cp_times": 3,
    "p_cp": 0.5,

    "do_inter_cp": True,
    "p_inter_cp": 1,
    # "do_match": False,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "p_scale": 0.2,

    "do_rotation": False,
    "degree": 180,
    "p_rot": 0.2,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "p_mirror": 0.8,

    "do_blurring": True,
    "blur_sigma": (1, 5),
    "p_blur": 0.1,

    # Not Implemented
    # "do_additive_brightness": False,
    # "additive_brightness_p_per_sample": 0.15,
    # "additive_brightness_p_per_channel": 0.5,
    # "additive_brightness_mu": 0.0,
    # "additive_brightness_sigma": 0.1,
}

def generate_prob_function(mask_shape):
    sigma = np.random.uniform(3,15)
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # Gaussian filter
    # this taks some time
    a_2 = gaussian_filter(a, sigma=sigma)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    return a

# first generate 5*200*200*200

def get_texture(mask_shape):
    # get the prob function
    a = generate_prob_function(mask_shape) 

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # if a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter

    # Gaussian filter
    if np.random.uniform() < 0.7:
        sigma_b = np.random.uniform(3, 5)
    else:
        sigma_b = np.random.uniform(5, 8)

    # this takes some time
    b2 = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b2, 0, 1) # 目前是0-1区间
    
    return Bj


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b, 0, 1) # 目前是0-1区间

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan, method = "generation"):
    if method == "CP": 
        mask_scan=torch.from_numpy(mask_scan).permute(1,2,0).detach().numpy() # 以下代码默认输入维度(h,w,d) 
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5,5), dtype=np.uint8)
    # print(type(liver_mask),type(liver_mask))
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    # print('xy',xyz)

    
    if method == "CP":
        xyz.insert(0,z)
        # print('zxy',xyz)
        potential_points = xyz
    else: 
        xyz.append(z)
        potential_points = xyz

    # print('pos',potential_points)
    return potential_points

# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type):

    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    # texture_map = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.float16)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'large':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == "mix":
        # tiny
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

        # small
        num_tumor = random.randint(5,10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
        
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture
            
        # medium
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # large
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    # texture_map = texture_map[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    geo_mask = (geo_mask * mask_scan) >=1
    
    return geo_mask


def get_tumor(volume_scan, mask_scan, tumor_type, texture):
    geo_mask = get_fixed_geo(mask_scan, tumor_type)
    sigma      = np.random.uniform(1, 2)
    difference = np.random.uniform(65, 145)

    # blur the boundary
    geo_blur = gaussian_filter(geo_mask*1.0, sigma)
    abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan
    
    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask
    # print('hha',np.unique(abnormally_mask))

    return abnormally_full, abnormally_mask

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start+1), min(mask_scan.shape[0], x_end-1)
    y_start, y_end = max(0, y_start+1), min(mask_scan.shape[1], y_end-1)
    z_start, z_end = max(0, z_start+1), min(mask_scan.shape[2], z_end-1)

    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask   = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    if TUMOR_LABEL not in liver_mask:
        # print('tumor generation')

    # input texture shape: 420, 300, 320
    # we need to cut it into the shape of liver_mask
    # for examples, the liver_mask.shape = 286, 173, 46; we should change the texture shape
        x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start
        start_x = random.randint(0, texture.shape[0] - x_length - 1) # random select the start point, -1 is to avoid boundary check
        start_y = random.randint(0, texture.shape[1] - y_length - 1) 
        start_z = random.randint(0, texture.shape[2] - z_length - 1) 
        cut_texture = texture[start_x:start_x+x_length, start_y:start_y+y_length, start_z:start_z+z_length]

        liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture)
    else:
        # print('tumor CP')
        liver_volume = torch.from_numpy(liver_volume).permute(2,0,1).detach().numpy()
        liver_mask = torch.from_numpy(liver_mask).permute(2,0,1).detach().numpy()
        liver_volume, liver_mask = aug_one_pair(liver_volume, liver_mask)
        liver_volume = torch.from_numpy(liver_volume).permute(1,2,0).detach().numpy()
        liver_mask = torch.from_numpy(liver_mask).permute(1,2,0).detach().numpy()

    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    return volume_scan, mask_scan


#############from tumor_CP############################

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

def paste_to( center, data_patch, seg_patch, tgt_data, tgt_seg):
    start_z, end_z, (ps_z, pe_z) = get_valid_center(center[0], data_patch.shape[0], tgt_data.shape[0])
    start_x, end_x, (ps_x, pe_x) = get_valid_center(center[1], data_patch.shape[1], tgt_data.shape[1])
    start_y, end_y, (ps_y, pe_y) = get_valid_center(center[2], data_patch.shape[2], tgt_data.shape[2])
    seg_patch = seg_patch[ps_z:pe_z, ps_x: pe_x, ps_y:pe_y]
    data_patch = data_patch[ps_z:pe_z, ps_x: pe_x, ps_y:pe_y]
    if seg_patch.shape[-1] != tgt_data[start_z:end_z, start_x:end_x, start_y:end_y].shape[-1]:
        end_z += 1
    # if cp_configs['do_inter_cp'] and cp_configs['do_match']:
    tgt_data[start_z:end_z, start_x:end_x, start_y:end_y]\
            = (seg_patch!=TUMOR_LABEL) * tgt_data[start_z:end_z, start_x:end_x, start_y:end_y] \
            + (seg_patch==TUMOR_LABEL) * (data_patch + np.mean(tgt_data[start_z:end_z, start_x:end_x, start_y:end_y]) - np.mean(data_patch))
    # else:
    #     tgt_data[start_z:end_z, start_x:end_x, start_y:end_y]\
    #             = (seg_patch!=TUMOR_LABEL) * tgt_data[start_z:end_z, start_x:end_x, start_y:end_y] \
    #             + (seg_patch==TUMOR_LABEL) * data_patch

    tgt_seg[start_z:end_z, start_x:end_x, start_y:end_y]\
            = (seg_patch!=TUMOR_LABEL) * tgt_seg[start_z:end_z, start_x:end_x, start_y:end_y] \
            + (seg_patch==TUMOR_LABEL) * seg_patch
    return tgt_data, tgt_seg


def aug_one_pair( liver_volume,liver_mask):
  
    labelmap, numregions = label(liver_mask == TUMOR_LABEL, return_num=True)
    if numregions == 0:
            return
    tumors = []
    for l in range(1, numregions + 1):
        bbox = get_bbox_from_mask(labelmap==l, pad = 0)
        cropped_data = crop_to_bbox(liver_volume, bbox)
        cropped_seg = crop_to_bbox(labelmap==l,bbox).astype(np.uint8)
        cropped_seg[cropped_seg>0] = TUMOR_LABEL
        tumors.append([cropped_data, cropped_seg])
    # print(len(tumors))
    tgt_data, tgt_seg = liver_volume, liver_mask
        
    for _ in range(cp_configs['cp_times']): # *self
        center = random_select(liver_mask,method = "CP")
        # Randomly choose a location for paste
        # Randomly choose a tumor from source
        tumor_index = random.choice(np.arange(len(tumors)))
        source_tumor = tumors[tumor_index]
        cropped_data, cropped_seg = source_tumor[0], source_tumor[1]
        # print('$$$$$$$$$$', np.sum(cropped_seg==TUMOR_LABEL))
        
        # crop_tumor_time = time.time()
        # print('crop_tumor_time', crop_tumor_time - aug_start_time)
        ##################################
      
        
        if cp_configs['do_rotation'] or cp_configs['do_scaling'] or cp_configs['do_elastic']:
            cropped_data, cropped_seg = do_spatial_augment(cropped_data, cropped_seg,
                                                            do_scaling=cp_configs['do_scaling'], p_scale=cp_configs['p_scale'],
                                                            do_rotation=cp_configs['do_rotation'], p_rot=cp_configs['p_rot'],
                                                            do_elastic=cp_configs['do_elastic'], p_elas=cp_configs['p_eldef'])
        
        # spatial_time = time.time()
        # print('spatial_time', spatial_time - crop_tumor_time)
        ##################################
            
        # if cp_configs['do_blurring'] and np.random.uniform() <= cp_configs['p_blur']:
        #     cropped_data, cropped_seg = do_blurring(cropped_data, cropped_seg, 
        #                                             blur_sigmma=cp_configs['blur_sigma'])
        
        # # blur_time = time.time()
        # # print('blur_time', blur_time - spatial_time)
        # ##################################
        
        # if cp_configs['do_mirror'] and np.random.uniform() <= cp_configs['p_mirror']:
        #     cropped_data, cropped_seg = do_flipping(cropped_data, cropped_seg)
        
        # # mirror_time = time.time()
        # # print('mirror_time', mirror_time - blur_time)
        # ##################################
        
        # if cp_configs['do_gamma'] and np.random.uniform() <= cp_configs['p_gamma']:
        #     cropped_data, cropped_seg = do_gamma(cropped_data, cropped_seg, 
        #                                             retain_stats=cp_configs['gamma_retain_stats'],
        #                                             gamma_range=cp_configs['gamma_range'])

        
        tgt_data, tgt_seg = paste_to(center, 
                                            cropped_data, cropped_seg, 
                                            tgt_data, tgt_seg)
      
    return tgt_data, tgt_seg

