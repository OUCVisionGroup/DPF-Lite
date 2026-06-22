import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import cv2

#==========================augmentation==========================
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def img_rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                                 borderValue=(0, 0, 0), )
    return rotated_img

def zoom(x, zx, zy, row_axis=0, col_axis=1):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]

    matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = cv2.warpAffine(x, matrix[:2, :], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                       borderValue=(0, 0, 0), )
    return x

def augmentation(img1, img2, img3):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot = random.random() < 0.3
    zo = random.random() < 0.3
    angle = random.random() * 180 - 90
    if hflip:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
        img3 = cv2.flip(img3, 1)
    if vflip:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)
        img3 = cv2.flip(img3, 0)
    if zo:
        zoom_range = (0.7, 1.3)
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img1 = zoom(img1, zx, zy)
        img2 = zoom(img2, zx, zy)
        img3 = zoom(img3, zx, zy)
    if rot:
        img1 = img_rotate(img1, angle)
        img2 = img_rotate(img2, angle)
        img3 = img_rotate(img3, angle)
    return img1, img2, img3
#==========================augmentation==========================


#==========================color transfer==========================
def get_mean_and_std(img):
    x_mean, x_std = cv2.meanStdDev(img)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

def get_s_mean_with_depth(s_mean, depth):
    count = depth.shape[0] * depth.shape[1]
    a = -0.05
    x, y, z = s_mean
    s_x = x * (1 - a * (1 - depth))
    s_y = y * (1 - a * (1 - depth))
    s_z = z * (1 - a * (1 - depth))
    s_x = s_x * (x / np.sum(s_x) * count)
    s_y = s_y * (y / np.sum(s_y) * count)
    s_z = s_z * (z / np.sum(s_z) * count)
    s_mean_depth = np.stack((s_x, s_y, s_z), axis=2)
    return s_mean_depth

def sample(mean, std):
    return np.random.normal(mean, std, size=1)[0]

def color_transfer(sc ,dep):
    sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
    s_mean, s_std = get_mean_and_std(sc)

    dep = dep / 255

    t_mean = np.array([97.0, 128.96, 112.25])
    t_std = np.array([50.49, 7.96, 24.33])
    # t_mean = np.array([sample(106.27, 36.76), sample(114.76, 29.69), sample(117.79, 31.58)])
    # t_std = np.array([sample(62.45, 15.33), sample(58.91, 12.04), sample(60.97, 12.03)])

    s_mean_depth = get_s_mean_with_depth(s_mean, dep)
    img_n = ((sc-s_mean_depth)*(t_std/(s_std+0.000000001)))+t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
    return dst
#==========================color transfer==========================

def pre_B_estimate(raw, device):
    bgl = np.zeros_like(raw)
    raw = np.transpose(raw, (2, 0, 1))

    for i in range(3):
        raw[i][raw[i] < 5] = 5
        raw[i][raw[i] > 250] = 250

    avg_B = np.mean(raw[0])
    std_B = np.std(raw[0])
    bgl_B = 1.13 * avg_B + 1.11 * std_B - 25.6

    avg_G = np.mean(raw[1])
    std_G = np.std(raw[1])
    bgl_G = 1.13 * avg_G + 1.11 * std_G - 25.6

    med_R = np.median(raw[2])
    bgl_R = 140 / (1 + 14.4 * np.exp(-0.034 * med_R))

    bgl[..., 0] = bgl_R
    bgl[..., 1] = bgl_G
    bgl[..., 2] = bgl_B

    bgl = torch.from_numpy(bgl / 255.0)
    return bgl.to(device, dtype=torch.float32).permute(2, 0, 1)

def preprocess(img1, img2, img3, device, isTrain):
    '''
    :param img1: raw data
    :param img2: ref data
    :param img3: depth map
    '''
    if isTrain:
        BL = pre_B_estimate(img1, device)
        # img1 = color_transfer(img1, img3)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = np.uint8((np.asarray(img1)))
        img2 = np.uint8((np.asarray(img2)))
        img3 = np.uint8((np.asarray(img3)))
        img1, img2, img3 = augmentation(img1, img2, img3)
    else:
        BL = pre_B_estimate(img1, device)
        # img1 = color_transfer(img1, img3)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = np.uint8((np.asarray(img1)))
        img2 = np.uint8((np.asarray(img2)))
        img3 = np.uint8((np.asarray(img3)))

    data1 = torch.from_numpy(img1 / 255.0)
    data2 = torch.from_numpy(img2 / 255.0)
    data3 = torch.from_numpy(img3 / 255.0)
    return (data1.to(device, dtype=torch.float32).permute(2, 0, 1),
            data2.to(device, dtype=torch.float32).permute(2, 0, 1),
            data3.to(device, dtype=torch.float32).unsqueeze(0), BL)

def populate_raw_list(raw_images_path):
    image_list_raw = glob.glob(raw_images_path + "/*.jpg")
    train_list = sorted(image_list_raw)

    return train_list

class UIEB_Dataset(Dataset):
    def __init__(self, device, raw_images_path=r'./UIEB/train/raw', Image_size=256, isTrain=True):
        self.raw_list = populate_raw_list(raw_images_path)
        raw_path = self.raw_list
        self.ref_list = [s.replace("raw", "ref") for s in raw_path]
        self.depth_list = [s.replace("raw", "raw_depth") for s in raw_path]
        self.size = Image_size
        self.isTrain = isTrain
        self.device = device

        print("Total image pairs:", len(self.raw_list))

    def __getitem__(self, index):
        data_raw_path = self.raw_list[index]
        file_name = data_raw_path.split('/')[-1].split('.')[0]
        data_raw = cv2.imread(data_raw_path)
        data_raw = cv2.resize(data_raw, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        data_ref_path = self.ref_list[index]
        data_ref = cv2.imread(data_ref_path)
        data_ref = cv2.resize(data_ref, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        data_depth_path = self.depth_list[index]
        data_depth = cv2.imread(data_depth_path, cv2.IMREAD_GRAYSCALE)
        data_depth = cv2.resize(data_depth, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        data_raw, data_ref, data_depth, BL = preprocess(data_raw, data_ref, data_depth, self.device, self.isTrain)

        return data_raw, data_ref, data_depth, BL, file_name

    def __len__(self):
        return len(self.raw_list)


