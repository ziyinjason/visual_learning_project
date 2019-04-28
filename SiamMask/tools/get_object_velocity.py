import numpy as np
import imageio
import pdb

def get_object_velocity_size(flow, state, M):
    H = state['im_h']
    W = state['im_w']

    x = np.linspace(0, W-1, W).astype(int)
    y = np.linspace(0, H-1, H).astype(int)
    mesh = np.meshgrid(y, x)
    mesh_x = mesh[1].reshape(1, W*H).T
    mesh_y = mesh[0].reshape(1, W*H).T
    pos = np.hstack((mesh_y,mesh_x, np.ones_like(mesh_x)))

    flow_back= np.dot(M, pos.T) - pos.T
    # pdb.set_trace()
    flow_back = flow_back.T.reshape((W, H, 3))[:,:,0:2]

    flow_back = flow_back.transpose((1,0,2))

    flow_object = flow - flow_back

    mask = state['mask']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    p = state['p']

    target_mask = (mask > p.seg_thr).astype(np.uint8)

    imageio.imwrite('mask_1.jpg',  target_mask*255)

    # pdb.set_trace()
    target_mask = np.tile(target_mask, (2, 1, 1))
    target_mask = target_mask.transpose((1, 2, 0))

    flow_object_segmented = target_mask * flow_object

    velocity = 2 * np.sum(flow_object_segmented, axis = (0,1)) / np.sum(target_mask)

    position = np.hstack((mesh_y,mesh_x)).reshape((W, H, 2))
    position = position.transpose((1,0,2))
    position_segmented = target_mask * position

    new_position = position_segmented + flow_object_segmented

    # pdb.set_trace()

    h = np.amax(new_position[:,:,0]) - np.amin(new_position[:,:,0])
    w = np.amax(new_position[:,:,1]) - np.amin(new_position[:,:,1])

    sz = np.array([w, h])

    return velocity, sz



