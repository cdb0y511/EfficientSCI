import numpy as np
import scipy.io
import cv2
import os
import einops
path = "/home/duo/datasets/NeRF/input_data"

def convertNerf(name):
    gt_dir = '/home/duo/datasets/NeRF/gt/' + name
    output_path = '/home/duo/ASources/EfficientSCI/test_datasets/nerf_data/' + name +'.mat'
    mask_path = '/home/duo/datasets/NeRF/input_data/' + name + '/mask_25.npy'
    output_path_mask = '/home/duo/ASources/EfficientSCI/test_datasets/mask/' + name +'.mat'
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
    else:
        mask_path = '/home/duo/datasets/NeRF/input_data/' + name + '/mask.npy'
        mask = np.load(mask_path)
    # Load mask and measurement data
    mask = np.load(mask_path)
    orig_bayer = []

    # Load ground truth images
    gt_images = []
    if os.path.exists(gt_dir):
        for filename in sorted(os.listdir(gt_dir)):
            if filename.endswith('.png'):
                img_path = os.path.join(gt_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                gt_images.append(img)
    else:
        return

    # Convert ground truth images to numpy array
    gt_images = np.array(gt_images)
    r = np.array([[1, 0], [0, 0]])
    g1 = np.array([[0, 1], [0, 0]])
    g2 = np.array([[0, 0], [1, 0]])
    b = np.array([[0, 0], [0, 1]])
    rgb2raw = np.zeros([3, mask.shape[1], mask.shape[2]])
    rgb2raw[0, :, :] = np.tile(r, (mask.shape[1] // 2, mask.shape[2] // 2))
    rgb2raw[1, :, :] = np.tile(g1, (mask.shape[1] // 2, mask.shape[2] // 2)) + np.tile(g2, (
        mask.shape[1] // 2, mask.shape[2] // 2))
    rgb2raw[2, :, :] = np.tile(b, (mask.shape[1] // 2, mask.shape[2] // 2))

    gt = []
    m_cr, m_h, m_w = mask.shape
    i_cr = len(gt_images)
    i_h, i_w, c = gt_images[0].shape
    assert m_cr == i_cr and m_h == i_h and m_w == i_w, "Image size does not match mask size! "
    meas = np.zeros_like(mask[0]).astype(float)
    for i, img in enumerate(gt_images):
        img = img.astype(np.float32) / 255
        img = einops.rearrange(img, "h w c->c h w")
        img = img[::-1, :, :]
        Y = np.sum(img * rgb2raw, axis=0)
        orig_bayer.append(Y)
        meas += np.multiply(mask[i, :, :].astype(float), Y)
    mask = np.transpose(mask,(1,2,0))
    gt_images = np.transpose(gt_images, (1, 2, 3, 0))
    meas = np.expand_dims(meas, axis=-1)
    orig_bayer = np.array(orig_bayer)
    orig_bayer = np.transpose(orig_bayer, (1, 2, 0))
    # Save data to .mat file
    scipy.io.savemat(output_path, {'Bayer_mode': 'rggb', 'mask_bayer': mask, 'meas_bayer': meas, 'orig': gt_images,
                                   'orig_bayer': orig_bayer}, do_compression=True)
    scipy.io.savemat(output_path_mask, {'mask': mask})
    print(f'Data saved to {output_path}')

if __name__=="__main__":
    names = os.listdir(path)
    for name in names:
        convertNerf(name)