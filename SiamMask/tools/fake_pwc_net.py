import numpy as np
import pdb

def fake_pwc_net_flow(video):

    if 'DAVIS' in video['image_files'][0].split('/'):
        data_set = video['image_files'][0].split('/')[-5]
    else:
        data_set = video['image_files'][0].split('/')[-3]

    # print(data_set)

    if 'DAVIS' in data_set:
        data_set = 'DAVIS'
        data_name = 'davis'
        name = video['name']
        path = '/home/jianingq/jianren/vot_rl/SiamMask/pwc_flow/' + data_set + '/' + name + '.npy'
    if 'VOT2018' in data_set:
        # data_name = 'vot_2018'
        name = video['name']
        path = '/home/jianingq/jianren/vot_rl/SiamMask/pwc_flow/' + data_set + '/' + name + '.npy'
    if 'VOT2016' in data_set:
        # data_name = 'vot_2016'
        name = video['name']
        path = '/home/jianingq/jianren/vot_rl/SiamMask/pwc_flow/' + data_set + '/' + name + '.npy'

    flow = np.load(path)
    return flow 

def load_M(dataset,video):

    if dataset=='DAVIS':
        video_name=video['name']
        #data_name = video['image_files'][0].split('/')
        file_name=video['image_files'][0].split('/')[-2]
        path = '/home/jianingq/jianren/vot_rl/SiamMask/pwc_flow/DAVIS_M/' + file_name + '.npy'
    if dataset=='VOT2018':
        video_name=video['name']
        file_name=video['image_files'][0].split('/')[-2]
        path = '/home/jianingq/jianren/vot_rl/SiamMask/pwc_flow/VOT2018_M/' + file_name + '.npy'
    if dataset=='VOT2016':
        video_name=video['name']
        file_name=video['image_files'][0].split('/')[-2]
        path = '/home/jianingq/jianren/vot_rl/SiamMask/pwc_flow/VOT2016_M/' + file_name + '.npy'
    M = np.load(path)
	
    return M