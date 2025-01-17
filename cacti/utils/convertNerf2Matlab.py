import numpy as np
import scipy.io
import cv2
import os
import einops
path = "/home/duo/datasets/NeRF/input_data"

def convertNerf(name):
    mask_path = '/home/duo/datasets/NeRF/input_data/' + name + '/mask_25.npy'
    meas_path = "/home/duo/datasets/NeRF/input_data/" + name + "/meas_25.npy"
    gt_dir = '/home/duo/datasets/NeRF/gt/' + name
    output_path = '/home/duo/ASources/EfficientSCI/test_datasets/nerf_data_rgb/' + name +'.mat'
    output_path_mask = '/home/duo/ASources/EfficientSCI/test_datasets/mask/' + name +'.mat'
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        meas = np.load(meas_path)
    else:
        mask_path = '/home/duo/datasets/NeRF/input_data/' + name + '/mask.npy'
        meas_path = "/home/duo/datasets/NeRF/input_data/" + name + "/meas.npy"
        mask = np.load(mask_path)
        meas = np.load(meas_path)
    # Load mask and measurement data
    gt_images = []
    if os.path.exists(gt_dir):
        for filename in sorted(os.listdir(gt_dir)):
            if filename.endswith('.png'):
                img_path = os.path.join(gt_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                gt_images.append(img)
    # Convert ground truth images to numpy array
        gt_images = np.array(gt_images)
        gt_images = np.transpose(gt_images, (1, 2, 3, 0))
    m_cr, m_h, m_w = mask.shape
    mask = np.transpose(mask,(1,2,0))
    # Save data to .mat file
    if os.path.exists(gt_dir):
        scipy.io.savemat(output_path, {'meas': meas, 'orig': gt_images}, do_compression=True)
    else:
        scipy.io.savemat(output_path, {'meas': meas}, do_compression=True)

    scipy.io.savemat(output_path_mask, {'mask': mask})
    print(f'Data saved to {output_path}')

if __name__=="__main__":
    names = os.listdir(path)
    for name in names:
        convertNerf(name)