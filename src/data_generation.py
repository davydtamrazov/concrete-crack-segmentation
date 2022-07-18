import os
import glob
import sys
import numpy as np
from itertools import product
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split

def generate_patches(img_path, gt_path, save_path, res=256, overlap=64, scale=1, lim=0.01):
    '''Split images into overlapping patches having a proportion 
    of positive class larger than a defined threshold.
    
    Args:
        img_path::str
            Path to the image
        gt_path::str
            Path to the corresponding ground truth
        save_path::str
            Path to save the split images
        res::int
            Resolution of the generated patches (default: 256)
        overlap::int
            Overlap of the generated patches (default: 64)
        scale::float
            Scale to scale down image resolution (default: 1)
        lim::float
            Threshold for the proportion of positive class (default: 0.01)
            
    Returns:
        None
    '''
    
    # Image name
    name = img_path.split('/')[-1].split('.')[0]
    
    # Print progress
    print(f"Processing image {name}") 
    
    # Read image and ground truth
    img = imread(img_path)
    gt = imread(gt_path, as_gray=True)
    
    # Check if paths exist
    if not os.path.exists(save_path+'/img'):
        os.makedirs(save_path+'/img')
    if not os.path.exists(save_path+'/gt'):
        os.makedirs(save_path+'/gt')
    
    # Check if image and ground truth are the same size
    if img.shape[0] != gt.shape[0] or img.shape[1] != gt.shape[1]: 
        print('ERROR: Image and ground truth dimensions are not the same.')
        sys.exit()

    # Rescale image and ground truth
    img_scaled = rescale(img, (scale, scale, 1))
    gt_scaled = rescale(gt, (scale, scale))[..., np.newaxis]
    
    h, w, c = img_scaled.shape

    # Get indices of overlapping windows
    h_ind = np.arange(0, h-res+1, res-overlap)
    w_ind= np.arange(0, w-res+1, res-overlap)
    
    # Resize image and ground truth to appropriate resolution
    if h-h_ind[-1] > res: hr=h_ind[-1]+res
    if w-w_ind[-1] > res: wr=w_ind[-1]+res
    
    # Resize image and ground truth
    if h!=hr or w!=wr:
        img_scaled = resize(img_scaled, (hr, wr, c), anti_aliasing=True)
        gt_scaled = resize(gt_scaled, (hr, wr, 1))
    
    # Loop through each window and save the image
    for s, (r,c) in enumerate(product(h_ind,w_ind)):
        if (np.sum(gt_scaled[r:r+res, c:c+res, :]==1)/(res**2)) > lim:

            # Crop patch out of image and ground truth
            img_crop = np.clip(img_scaled[r:r+res, c:c+res, :], -1, 1)
            gt_crop = np.clip(gt_scaled[r:r+res, c:c+res, :], -1, 1) >= 0.5
            
            imsave(f'{save_path}/img/{name}_patch{s}_scale{int(scale*100)}.jpg', 
                   img_as_ubyte(img_crop), check_contrast=False)
            imsave(f'{save_path}/gt/{name}_patch{s}_scale{int(scale*100)}.jpg', 
                   img_as_ubyte(gt_crop), check_contrast=False)
 
 
def get_file_list(path, file_format=".jpg"):
    '''Iterate through a folder and return a list of image paths 
    and ground truth paths.
    
    Args:
        path::str
            Path to the folder
        file_format::str
            File format of the files (default: ".jpg")
    
    Returns:
        file_list::[str]
            List of relevant files in the folder
    '''
    
    return sorted(glob.glob(path + '/*' + file_format))
               
def get_train_test_split(data, train_val_test_split, random_state=0):
    '''Split data into training, validation and test sets.
    
    Args:
        data::np.array()
            Data to be split
        train_val_test_split::[float]
            Proportion of data to be used for training, validation and test sets
    
    Returns:
        train::np.array()
            Training set
        val::np.array()
            Validation set
        test::np.array()
            Test set
    '''
    
    # Check if ratios sum to 1
    if sum(np.array(train_val_test_split)) != 1:
        print('ERROR: Training, validation and test ratios must sum to 1.')
        sys.exit()

    _, val_size, test_size = train_val_test_split
    train, val, test = data, [], []
    
    # Split into training and test sets  
    if test_size:
        train, test = train_test_split(data, test_size=test_size,
                                       random_state=random_state)
    
    # Split into training and validation sets
    if val_size:
        val_size = val_size/(len(train)/(len(train)+len(test)))
        train, val = train_test_split(train, test_size=val_size, 
                                    random_state=random_state)

    return train, val, test

    
if __name__ == '__main__':
    
    # Get list of files
    img_folder_path = './data/datasets/concrete/img'
    gt_folder_path = './data/datasets/concrete/gt'

    img_list = get_file_list(img_folder_path, "")
    gt_list = get_file_list(gt_folder_path, ".jpg")

    # Check if number of files is the same
    if len(img_list) != len(gt_list):
        print('ERROR: Number of images and ground truth files is not the same.')
        sys.exit()
    
    # Split data into training, validation and test sets
    ind = np.arange(len(img_list))
    train, val, test = get_train_test_split(ind, [0.6, 0.2, 0.2])
    
    # Loop through each image and save the patches
    for s in [0.25,0.5,0.75]:
        for i, n in enumerate(test):
            generate_patches(img_list[n], gt_list[n], './data/train', res=256, 
                            overlap=64, scale=s, lim=0.01)
            print(f"{i+1}/{len(img_list)}")
