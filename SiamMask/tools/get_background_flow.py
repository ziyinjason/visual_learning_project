import numpy as np
import cv2
import pdb

def pixel_flow(p, flow):
    num_sample, __ = np.shape(p)
    pixel_add_flow = []
    for i in range(num_sample):
        x = p[i, 1]
        y = p[i, 0]
        u, v = flow[y,x]
        pixel_add_flow += [[y+u, x+v, 1]]
    pixel_add_flow = np.array(pixel_add_flow)
    return pixel_add_flow

def get_background_flow(flow, H, W):

    x = np.linspace(0, W-1, W).astype(int)
    y = np.linspace(0, H-1, H).astype(int)
    mesh = np.meshgrid(y, x)
    mesh_x = mesh[1].reshape(1, W*H).T
    mesh_y = mesh[0].reshape(1, W*H).T
    pos = np.hstack((mesh_y,mesh_x))

    pos_next = pixel_flow(pos, flow).astype(int)
    M, _ = cv2.findHomography(pos, pos_next, cv2.RANSAC,5.0)

    pos_extend = np.hstack((pos, np.ones_like(mesh_x)))
    flow_back= np.dot(M, pos_extend.T) - pos_extend.T

    flow_back = flow_back.T.reshape((H, W, 3))

    return M, flow_back