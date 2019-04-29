import numpy as np
import cv2
import pdb
import imageio
from tools.draw_box import draw_bbox

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


def get_object_velocity_size(flow, state, M,ob_flow_only=False): # f,obj_id,im,
    H = state['im_h']
    W = state['im_w']
    x = np.arange(W)
    y = np.arange(H)
    source = np.meshgrid(x, y)
    source = np.array(source).transpose(1, 2, 0)
    # source_x = source[:, :, 0].reshape(1, W * H).T
    # source_y = source[:, :, 1].reshape(1, W * H).T
    # source_pos = np.hstack((source_x, source_y, np.ones_like(source_x)))
    # bg_flow = np.dot(M, source_pos.T) - source_pos.T
    # bg_flow = bg_flow.reshape(3, H, W).transpose(1, 2, 0)[:, :, :-1]

    # ob_flow = flow - bg_flow

    mask = state['mask']
    p = state['p']

    target_mask = (mask > p.seg_thr).astype(np.uint8)
    target_mask = np.tile(target_mask.reshape(H, W, 1), (1, 1, 2))

    # flow_object_segmented = target_mask * ob_flow

    # x,y
    # target_sum=np.sum(target_mask)
    target_sum=target_mask.sum()
    #pdb.set_trace()
    #if len(new_position[non_zero_position[:,:,0],0])==0:
    if target_sum==0:
        velocity=np.array([0.00001,0.00001])
        return velocity,state['target_sz']
        #return velocity,state['target_sz'],np.array([0,0])

    velocity = 2 * np.sum(target_mask * flow, axis=(0, 1)) / target_sum
    # velocity=2*np.array([flow_object_segmented[:,:,0].sum(),flow_object_segmented[:,:,1].sum()])/target_sum

    #bg_flow_with_mask=target_mask*bg_flow
    #mean_bg_speed = 2*np.sum( target_mask*bg_flow,axis=(0,1))/target_sum
    # mean_bg_speed = 2*np.array([bg_flow_with_mask[:,:,0].sum(),bg_flow_with_mask[:,:,1].sum()])/target_sum
    new_position = target_mask * (flow + source)
    non_zero_position=new_position>0
    # if f==86 and obj_id==2:
    #     xtl,ytl= np.round(state['target_pos']-state['target_sz']/2).astype('int')
    #     xbr,ybr= np.round(state['target_pos']+state['target_sz']/2).astype('int')
    #     im_bbox=draw_bbox(im,[xtl,ytl,xbr,ybr], color=[0,0,255], width=3)
    #     imageio.imwrite('image/bbx_image_'+str(f)+'.jpg',im_bbox)
    #     pdb.set_trace()
    
    non_zero_position=new_position>0

    if ob_flow_only:
        if len(new_position[non_zero_position[:,:,0],0])==0:
            return velocity,state['target_sz']

        w = np.amax(new_position[:, :, 0])-np.amin(new_position[non_zero_position[:,:,0],0])
        h = np.amax(new_position[:, :, 1])-np.amin(new_position[non_zero_position[:,:,1],1])
        sz = np.array([w, h])

        if w==0 or h==0:
            sz=state['target_sz']

        return velocity,sz

    if len(new_position[non_zero_position[:,:,0],0])==0:
        velocity=np.array([0.00001,0.00001])
        return velocity,state['target_sz'],np.array([0,0])

    w = np.amax(new_position[:, :, 0]) - np.amin(new_position[non_zero_position[:,:,0],0])
    h = np.amax(new_position[:, :, 1]) - np.amin(new_position[non_zero_position[:,:,1],1])



    sz = np.array([w, h])
    
    return velocity, sz, mean_bg_speed
