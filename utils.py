import os
import sys
import numpy as np
import skimage as sk
import random
import cv2
import pickle
import math

from tqdm import tqdm
from itertools import chain

from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.io import imread, imshow
from skimage.transform import resize



from scipy.spatial.transform import Rotation
import re

def loadAndSplitRawData(disk_path, hsize, wsize, lenght, total_style, train_style, valid_style):    
    sys.stdout.flush()
    nbImg = len(os.listdir(disk_path))
    if (lenght == -1 or lenght > nbImg):
        lenght = nbImg
    
    x_train_lenght = int(lenght*(len(train_style)/total_style))
    x_valid_lenght = int(lenght*(len(valid_style)/total_style))
    
    x_train = np.zeros((x_train_lenght, hsize, wsize, 3), dtype=np.uint8)
    x_valid = np.zeros((x_valid_lenght, hsize, wsize, 3), dtype=np.uint8)

    n_train = 0    
    n_valid = 0            
    for n, img in enumerate(tqdm(os.listdir(disk_path))):
        if n == lenght:
            break            
        path = os.path.join(disk_path,img)
        if path.endswith(".png"):
            img = imread(path)                
            #print(path)
            #img = resize(img, (hsize, wsize, 3), mode='constant', preserve_range=True).astype(np.uint8)

            #background
            mask = np.zeros(img.shape[:2], np.uint8)
            mask = cv2.circle(mask, (int(img.shape[1]/2), int(img.shape[0]/2)), int(img.shape[0]/3), 255, -1)
            masked = cv2.bitwise_and(img, img, mask=mask)
            
            masked = cv2.resize(masked, (wsize, hsize), interpolation = cv2.INTER_AREA)
        
            if any([x in path for x in valid_style]):
                x_valid[n_valid] = masked
                n_valid+=1
            elif any([x in path for x in train_style]):    
                x_train[n_train] = masked
                n_train+=1
    
    return x_train, x_valid   

 
def loadPoseDataDict(disk_path, dofs, lenght = -1, duplicate = 1, ts = [1,1,1]):
    pose_dict = pickle.load(open(disk_path, 'rb'))
    nbItems = len(pose_dict)    
    
    if (lenght == -1 or lenght > nbItems):
        lenght = nbItems*duplicate
    
    x = np.zeros((lenght, dofs), dtype=np.float)
    dictKey = np.zeros(lenght)
    
    for n, item in enumerate(pose_dict):
        pose = pose_dict[item][1]

        # TOFIX with tanh: HACK FOR LAZY NORMALIZATION
        pose[3] = pose[3]/ts[0]
        pose[4] = pose[4]/ts[1]
        pose[5] = pose[5]/ts[2]

        
        r = np.zeros(3, dtype=np.float)
        t = np.zeros(3, dtype=np.float)

        r[0] = pose[0]
        r[1] = pose[1]
        r[2] = pose[2]
        
        t[0] = pose[3]
        t[1] = pose[4]
        t[2] = pose[5]
                 
        for d in range(duplicate):
            x[n*duplicate+d] = pose
            
            # keep key to check if all data were loaded in order
            dictKey[n*duplicate+d] = pose_dict[item][0]
              
    return x


def comparePoses(y, p, ts = [1,1,1]):
    
    r_pred = [p[0], p[1], p[2]]
    r_gt = [y[0], y[1], y[2]]
    
    # to Euler
    r_pred = Rotation.from_rotvec(r_pred).as_euler('xyz', degrees=True)
    r_gt = Rotation.from_rotvec(r_gt).as_euler('xyz', degrees=True)
    
    # normalization if needed
    t_pred = [p[3]*ts[0], p[4]*ts[1], p[5]*ts[2]]
    t_gt = [y[3]*ts[0], y[4]*ts[1], y[5]*ts[2]]
    
    # return translation and rotation error    
    return [math.dist(t_pred,t_gt), math.dist(r_pred,r_gt)]

def toMV(p, ts = [1,1,1]):
    
    r = [p[0], p[1], p[2]]
    
    # to matrix
    r = Rotation.from_rotvec(r).as_matrix()
    
    # normalization
    t = [p[3]*ts[0], p[4]*ts[1], p[5]*ts[2]]
    
    mv = np.array([
        [ r[0][0], r[0][1], r[0][2],  t[0]],
        [ r[1][0], r[1][1], r[1][2],  t[1]],
        [ r[2][0], r[2][1], r[2][2],  t[2]],
        [ 0.0,  0.0,  0.0,  1.0]
    ]) 
            
    # return translation and rotation error    
    return mv


def compute_ADD(y, p, objFilename, ts = [1,1,1]):
    
    v1 = [] 
    v1.append(y[0])
    v1.append(y[1])
    v1.append(y[2])
    
    t1 = []
    t1.append(y[3]*ts[0])
    t1.append(y[4]*ts[1])
    t1.append(y[5]*ts[2])

    v2 = [] 
    v2.append(p[0][0])
    v2.append(p[0][1])
    v2.append(p[0][2])
    
    t2 = []
    t2.append(p[0][3]*ts[0])
    t2.append(p[0][4]*ts[1])
    t2.append(p[0][5]*ts[2])
    
    reComp = re.compile("(?<=^)(v |vn |vt |f )(.*)(?=$)", re.MULTILINE)
    with open(objFilename) as f:
        data = [txt.group() for txt in reComp.finditer(f.read())]

    v_arr = []
    for line in data:
        tokens = line.split(' ')
        if tokens[0] == 'v':
            v_arr.append([float(c) for c in tokens[1:]])    
      
    r1 = (Rotation.from_rotvec(v1)).as_matrix()
    r2 = (Rotation.from_rotvec(v2)).as_matrix()
    
    dist = 0
    for pt in v_arr:
        pt1 = np.dot(r1, pt) + t1
        pt2 = np.dot(r2, pt) + t2
            
        dist += np.linalg.norm(pt1 - pt2)
        
    err = dist/len(v_arr)  
    
    
    return err


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
