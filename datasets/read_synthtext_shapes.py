import scipy.io as sio
import cv2
import numpy as np
from tqdm import tqdm
import pickle

data_dir = '/home/mobile/Downloads/SynthText/'

print("reading gt.mat file...")
gt = sio.loadmat(data_dir + 'gt.mat')

with open('image_size.pkl', 'w') as f:
    dst = {}
    num_of_file = gt['imnames'].shape[1]
    print("read image size...")
    for idx_of_img in tqdm(xrange(num_of_file)):
        image_file_name = str(gt['imnames'][:,idx_of_img][0][0])
        path = data_dir + image_file_name
        height, width, depth = cv2.imread(path).shape
        dst[image_file_name] = [height, width, depth]
    
    pickle.dump(dst, f, pickle.HIGHEST_PROTOCOL)
    print("saved the size into image_size.pkl file.")
    