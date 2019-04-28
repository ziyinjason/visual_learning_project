import sys
import numpy as np
import cv2
from scipy import spatial

sys.path.append('/home/jianingq/PWC-Net/PyTorch')
from jianren_script import compute_flow

import pdb
from IPython import embed

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

def get_velocity(mask, M, flow, state):
    p = state['p']
    mask = state['mask']

    H = mask.shape[0]
    W = mask.shape[1]
    x = np.arange(W)
    y = np.arange(H)
    source = np.meshgrid(x, y)
    source = np.array(source).transpose(1, 2, 0)
    source = 
    source_x = source[:, :, 0].reshape(1, W * H).T
    source_y = source[:, :, 1].reshape(1, W * H).T
    source_pos = np.hstack((source_x, source_y, np.ones_like(source_x)))

    flow_flatten = flow.reshape((H*W, -1))




def compute_velocity_on_ref(im_ref, im_cur, mask_ref, state):
    flow = compute_flow(im_cur, im_ref)
    M = get_homography(flow)

    embed()

    velocity = get_velocity(mask_ref, M, flow, state)

def update_with_homography(M, size, pos):

    size = size.reshape(-1, 1)
    pos = pos.reshape(-1, 1)
    tl_point = pos - size / 2
    br_point = pos + size / 2
    prev_points = np.concatenate((np.concatenate(
        (pos, tl_point, br_point, np.array([tl_point[0], br_point[1]]).reshape(
            -1, 1), np.array([br_point[0], tl_point[1]]).reshape(-1, 1)),
        axis=1).T, np.ones((5, 1))),
                                 axis=1).T
    updated_points = np.dot(M, prev_points)
    updated_pos = updated_points[:2, 0]
    updated_size = np.amax(
        updated_points[:2, 1:], axis=1) - np.amin(
            updated_points[:2, 1:], axis=1)
    return updated_pos, updated_size
