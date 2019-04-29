import os
import numpy as np
import cv2
import sys
from scipy.ndimage import imread

import imageio

sys.path.append('/home/jianingq/PWC-Net/PyTorch')
from jianren_script import compute_flow



root = '/home/jianingq/Workspace/vot/SiamRPN/sequences/pedestrian1/color/'

def get_homography(flow):

    H = flow.shape[0]
    W = flow.shape[1]
    x = np.arange(W)
    y = np.arange(H)
    source = np.meshgrid(x, y)
    source = np.array(source).transpose(1, 2, 0)
    source_x = source[:, :, 0].reshape(1, W * H).T
    source_y = source[:, :, 1].reshape(1, W * H).T
    target = source + flow
    target_x = target[:, :, 0].reshape(1, W * H).T
    target_y = target[:, :, 1].reshape(1, W * H).T
    source_pos = np.hstack((source_x, source_y))
    target_pos = np.hstack((target_x, target_y))
    M, _ = cv2.findHomography(
        source_pos, target_pos, cv2.RANSAC, 3.0, maxIters=100)
    return M

def main():
    cur_num = "%08d" % 25
    next_num = "%08d" % 27

    cur_img = imread(root + cur_num + '.jpg')
    next_img = imread(root + next_num + '.jpg')

    print('start computing flow!')

    flow = compute_flow(cur_img, next_img)

    print('start computing M!')

    M = get_homography(flow)

    print('finished!')

    M_inv = np.linalg.inv(M)

    print(M_inv)

    mapped_img = cv2.warpPerspective(next_img, M_inv, (cur_img.shape[1],cur_img.shape[0]))

    imageio('mapped_img' + next_num + '.jpg', mapped_img)

if __name__ == "__main__":
    main()

