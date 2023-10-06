import os
os.environ["CDF_LIB"] = "/is/cluster/scratch/stripathi/data/cdf37_1-dist/src/lib"

import cv2
import pandas as pd
# import json
# import glob
# import h5py
import torch
# import trimesh
import numpy as np
import pickle as pkl
# from xml.dom import minidom
# import xml.etree.ElementTree as ET
from tqdm import tqdm
from spacepy import pycdf
# from .read_openpose import read_openpose
import sys
sys.path.append('../../')
# from models import hmr, SMPL
# import config
# import constants
import argparse

# import shutil

import smplx
import pytorch3d.transforms as p3dt

from utils.geometry import batch_rodrigues, batch_rot2aa, ea2rm
# from vis_utils.world_vis import overlay_mesh, vis_smpl_with_ground


model_type = 'smplx'
model_folder = '/ps/project/common/smplifyx/models/'
body_model_params = dict(model_path=model_folder,
                         model_type=model_type,
                         create_global_orient=True,
                         create_body_pose=True,
                         create_betas=True,
                         num_betas=10,
                         create_left_hand_pose=True,
                         create_right_hand_pose=True,
                         create_expression=True,
                         create_jaw_pose=True,
                         create_leye_pose=True,
                         create_reye_pose=True,
                         create_transl=True,
                         use_pca=False)
body_model = smplx.create(gender='neutral', **body_model_params).to('cuda')

def rich_extract(img_dataset_path, out_path, scene_indx, split=None, vis_path=None, visualize=False, downsample_factor=1):
    
    # structs we use
    imgnames_ = []
    # scales_, centers_, parts_, Ss_, Ss_world_, openposes_ = [], [], [], [], [], []
    poses_, shapes_, transls_ = [], [], []
    # poses_world_, transls_world_, cams_r_, cams_t_ = [], [], [], []
    cams_k_ = []
    # ground_offset_ = []
    # in_bos_label_ = []
    contact_label_ = []
    scene_seg_, part_seg_ = [], []

    # # seqs in validation set
    # if split == 'val':
    #     seq_list = ['2021-06-15_Multi_IOI_ID_00176_Yoga2',
    #                 '2021-06-15_Multi_IOI_ID_00228_Yoga1',
    #                 '2021-06-15_Multi_IOI_ID_03588_Yoga1',
    #                 '2021-06-15_Multi_IOI_ID_00176_Yoga1']

    # # seqs in testing set
    # if split == 'test':
    #     seq_list = ['2021-06-15_Multi_IOI_ID_00186_Yoga1',
    #                 '2021-06-15_Multi_IOI_ID_03588_Yoga2',
    #                 'MultiIOI_201019_ID03581_parkingLot_Calibration06_Settings06_PushUp__2',
    #                 'Multi-IOI_ID00227_Scene_ParkingLot_Calibration_03_CameraSettings_4_pushup_1']
    
    scenes = sorted(os.listdir('/ps/project/datasets/RICH/rich_toolkit/data/images/test'))
    # scene = scenes[scene_indx]
    scene = 'ParkingLot2_009_impro5'
    
    for scene_name in [scene]:
    # for scene_name in ['LectureHall_003_wipingchairs1']:
        out_file = os.path.join(out_path, f'rich_{scene_name}_smplx.npz')
        if os.path.exists(out_file): return
        print(scene_name)
        for i, fl in tqdm(enumerate(sorted(os.listdir(os.path.join(img_dataset_path, 'images', split)))), dynamic_ncols=True):
            if not scene_name in fl: continue

            ind = fl.index('cam')
            location = fl[:ind-1]

            cam_num = fl[ind:ind+6]

            img = fl[ind+7:-3] + 'jpeg'

            imgname = os.path.join(location, cam_num, img)

            mask_name = fl
            sp = mask_name.split('_')
            indx = mask_name.index('cam')
            st = mask_name[indx-1:indx+7]
            mask_name = mask_name.replace(st, '/')
            mask_name = mask_name[:-7]
            new_p = mask_name.split('/')
            mask_name = new_p[0] + '/' + new_p[1] + '/' + sp[1] + '.pkl'
            mask_path = os.path.join(img_dataset_path, 'labels', split, mask_name)
            df = pd.read_pickle(mask_path)
            mask = df['contact'] 

            scene_path = os.path.join('/ps/scratch/ps_shared/stripathi/deco/4agniv/rich/seg_masks_new', split, fl[:-3] + 'png')

            part_path = os.path.join('/ps/scratch/ps_shared/stripathi/deco/4agniv/rich/part_masks_new', split, fl[:-3] + 'png')

            dataset_path = '/ps/project/datasets/RICH'

            ind = fl.index('cam')
            frame_id = fl[:ind-1]
            location = frame_id.split('_')[0]

            if location == 'LectureHall':
                if 'chair' in frame_id:
                    cam2world_location = location + '_' + 'chair'
                else:
                    cam2world_location = location + '_' + 'yoga'  
            else:
                cam2world_location = location        

            img_num = fl.split('_')[-2]

            cam_num = int(fl.split('_')[-1][:2])

            # get ioi2scan transformation per sequence
            ioi2scan_fn = os.path.join(dataset_path, 'website_release/multicam2world', cam2world_location + '_multicam2world.json')

            try:
                camera_fn = os.path.join(dataset_path, 'rich_toolkit/data/scan_calibration', location, f'calibration/{cam_num:03d}.xml')
                focal_length_x, focal_length_y, camC, camR, camT, _, _, _ = extract_cam_param_xml(camera_fn)
            except:
                print(f'camera calibration file not found: {camera_fn}')
                continue

            # print('X: ', focal_length_x)
            # print('Y: ', focal_length_y)
            
            # path to smpl params
            smplx_param = os.path.join(dataset_path, 'rich_toolkit/data/bodies', split, frame_id, str(img_num), frame_id.split('_')[1] + '.pkl')

            # # path to GT bounding boxes
            # bbox_path = os.path.join(dataset_path, 'preprocessed', split, frame_id, img_num, frame_id.split('_')[1], 'bbox_refine', f'{img_num}_{cam_num:02d}.json')
            # # path with 2D openpose keypoints
            # openpose_path = os.path.join(dataset_path, 'preprocessed', split, frame_id, img_num, frame_id.split('_')[1], 'keypoints_refine', f'{img_num}_{str(cam_num).zfill(2)}_keypoints.json')
            # # path to image crops
            # img_path = os.path.join(dataset_path, 'preprocessed', split, frame_id, img_num, frame_id.split('_')[1], 'images_refine', f'{img_num}_{cam_num:02d}.png')

            # # bbox file
            # try:
            #     with open(bbox_path, 'r') as f:
            #         bbox_dict = json.load(f)
            # except:
            #     print(f'bbox file not found: {bbox_path}')
            #     continue

            # # read GT bounding box
            # x1_ul = bbox_dict['x1'] // downsample_factor
            # y1_ul = bbox_dict['y1'] // downsample_factor
            # x2_br = bbox_dict['x2'] // downsample_factor
            # y2_br = bbox_dict['y2'] // downsample_factor
            # bbox = np.array([x1_ul, y1_ul, x2_br, y2_br])
            # center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            # scale = 0.9 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

            # get smpl parameters
            ## body resides in multi-ioi coordidate, where camera 0 is world zero.
            with open(smplx_param, 'rb') as f:
                body_params = pkl.load(f)
                # in ioi coordinates: cam 0
                beta = body_params['betas']
                pose_aa = body_params['body_pose']
                pose_rotmat = p3dt.axis_angle_to_matrix(torch.FloatTensor(pose_aa.reshape(-1,3))).numpy()

                transl = body_params['transl']
                global_orient = body_params['global_orient']
                global_orient = p3dt.axis_angle_to_matrix(torch.FloatTensor(global_orient.reshape(-1,3))).numpy()

            smpl_body_cam0 = body_model(betas=torch.FloatTensor(beta).to('cuda')) # canonical body with shape
            vertices_cam0 = smpl_body_cam0.vertices.detach().cpu().numpy().squeeze()
            joints_cam0 = smpl_body_cam0.joints.detach().cpu().numpy()
            pelvis_cam0 = joints_cam0[:, 0, :]

            # ## rigid transformation between multi-ioi and Leica scan (world)
            # with open(ioi2scan_fn, 'r') as f:
            #     ioi2scan_dict = json.load(f)
            #     R_ioi2world = np.array(ioi2scan_dict['R']) # Note: R is transposed
            #     t_ioi2world= np.array(ioi2scan_dict['t']).reshape(1, 3)     

            # # # get SMPL params in world coordinates
            # # # import ipdb; ipdb.set_trace()
            # global_orient_world = np.matmul(R_ioi2world.T, global_orient)
            # transl_world = np.matmul((pelvis_cam0+transl), R_ioi2world) + t_ioi2world - pelvis_cam0 # right multiplication to avoid transpose
            # full_pose_rotmat_world = np.concatenate((global_orient_world, pose_rotmat), axis=0).squeeze()
            # theta_world = batch_rot2aa(torch.FloatTensor(full_pose_rotmat_world)).reshape(-1, 66).cpu().numpy()

            # smpl_body_world = body_model(betas=torch.FloatTensor(beta).to('cuda'),
            #                             body_pose=torch.FloatTensor(theta_world[:, 3:]).to('cuda'),
            #                             transl=torch.FloatTensor(transl_world).to('cuda'),
            #                             global_orient=torch.FloatTensor(theta_world[:, :3]).to('cuda'))
            # vertices_world = smpl_body_world.vertices.detach().cpu().numpy().squeeze()
            # joints3d_world = smpl_body_world.joints[:, 25:, :].detach().cpu().numpy().squeeze()

            # # get SMPL params in camera coordinates
            global_orient_cam = np.matmul(np.array(camR), global_orient)
            transl_cam = np.matmul(camR, (pelvis_cam0 + transl).T).T + camT - pelvis_cam0
            full_pose_rotmat_cam = np.concatenate((global_orient_cam, pose_rotmat), axis=0).squeeze()
            theta_cam = batch_rot2aa(torch.FloatTensor(full_pose_rotmat_cam)).reshape(-1, 66).cpu().numpy()

            # read GT 2D keypoints
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = focal_length_x / downsample_factor
            K[1, 1] = focal_length_y / downsample_factor
            K[:2, 2:] = camC.T / downsample_factor

            # # get openpose 2D keypoints
            # try:
            #     with open(openpose_path, 'r') as f:
            #         openpose = json.load(f)
            #     openpose = np.array(openpose['people'][0]['pose_keypoints_2d']).reshape([-1, 3])
            # except:
            #     print(f'No openpose !! Missing {openpose_path}')
            #     continue

            # get camera parameters wrt to scan
            # R_worldtocam = np.matmul(camR, R_ioi2world) # Note: R_ioi2world is transposed
            # T_worldtocam = -t_ioi2world + camT

            # ground offset
            # ground_offset = ground_eq[2]

            # store data
            jpg_img_path = os.path.join('/ps/project/datasets/RICH_JPG', split, imgname)
            bmp_img_path = jpg_img_path.replace('/ps/project/datasets/RICH_JPG', '/ps/project/datasets/RICH/rich_toolkit/data/images')
            bmp_img_path = bmp_img_path.replace('.jpeg', '.bmp')
            if not os.path.exists(bmp_img_path):
                bmp_img_path = bmp_img_path.replace('.bmp', '.png')
            imgnames_.append(bmp_img_path)
            contact_label_.append(mask)
            scene_seg_.append(scene_path)
            part_seg_.append(part_path)
            # centers_.append(center)
            # scales_.append(scale)
            # openposes_.append(openpose)
            poses_.append(theta_cam.squeeze())
            transls_.append(transl_cam.squeeze())
            # poses_world_.append(theta_world.squeeze())
            # transls_world_.append(transl_world.squeeze())
            shapes_.append(beta.squeeze())
            # cams_r_.append(R_worldtocam.tolist())
            # # Todo: note that T_worldtocam here is (1,3) whereas in h36m T_worldtocam is (1,3)
            # cams_t_.append(T_worldtocam.tolist())
            cams_k_.append(K.tolist())
            # ground_offset_.append(ground_offset)

        # for seq_i in tqdm(seq_list):
        #     print(f'Processing sequence: {seq_i}')

        #     # path with GT bounding boxes
        #     params_path = os.path.join(dataset_path, seq_i, 'params')

        #     # path to metadata for files
        #     md_path = os.path.join(dataset_path, seq_i, 'data')

        #     # glob all folders in params path
        #     frame_param_paths = sorted(glob.glob(os.path.join(params_path, '*')))
        #     frame_param_paths = [p for p in frame_param_paths if '.yaml' not in p]

        #     # get ioi2scan transformation per sequence
        #     ioi2scan_fn = os.path.join(dataset_path, seq_i, 'cam2scan.json')

        #     ## ground resides in Leica scan coordinate, which is (roughly) axis aligned.
        #     # ground_mesh = trimesh.load(os.path.join(dataset_path, seq_i, 'ground_mesh.ply'), process=False)
        #     # ground_eq = np.mean(ground_mesh.vertices, axis=0)

        #     # list all files in the folder
        #     cam_files = os.listdir(os.path.join(dataset_path, seq_i, f'calibration'))
        #     cam_list = sorted([int(os.path.splitext(f)[0]) for f in cam_files if '.xml' in f])

        #     # if split == 'val':
        #     #     cam_list = cam_list[1:] # remove first camera in val
        #     for cam_num in cam_list[:1]:
        #         camera_fn = os.path.join(dataset_path, seq_i, f'calibration/{cam_num:03d}.xml')
        #         focal_length_x, focal_length_y, camC, camR, camT, _, _, _ = extract_cam_param_xml(camera_fn)

        #         for frame_param_path in tqdm(frame_param_paths):
        #             frame_id = os.path.basename(frame_param_path)
        #             frame_num = int(frame_id)

        #             # path to smpl params
        #             try:
        #                 smplx_param = os.path.join(frame_param_path, '00', 'results/000.pkl')
        #             except:
        #                 import ipdb; ipdb.set_trace()

        #             # path to GT bounding boxes
        #             bbox_path = os.path.join(md_path, frame_id, '00', 'bbox_refine', f'{frame_id}_{cam_num:02d}.json')
        #             # path with 2D openpose keypoints
        #             openpose_path = os.path.join(md_path, frame_id, '00', 'keypoints_refine', f'{frame_id}_{str(cam_num).zfill(2)}_keypoints.json')
        #             # path to image crops
        #             if downsample_factor == 1:
        #                 img_path = os.path.join(md_path, frame_id, '00', 'images_orig', f'{frame_id}_{cam_num:02d}.png')
        #             else:
        #                 img_path = os.path.join(md_path, frame_id, '00', 'images_orig_720p', f'{frame_id}_{cam_num:02d}.png')

        #             if not os.path.isfile(img_path):
        #                 print(f'image not found: {img_path}')
        #                 continue
        #                 # raise FileNotFoundError

        #             # bbox file
        #             try:
        #                 with open(bbox_path, 'r') as f:
        #                     bbox_dict = json.load(f)
        #             except:
        #                 print(f'bbox file not found: {bbox_path}')
        #                 continue
        #             # read GT bounding box
        #             x1_ul = bbox_dict['x1'] // downsample_factor
        #             y1_ul = bbox_dict['y1'] // downsample_factor
        #             x2_br = bbox_dict['x2'] // downsample_factor
        #             y2_br = bbox_dict['y2'] // downsample_factor
        #             bbox = np.array([x1_ul, y1_ul, x2_br, y2_br])
        #             center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        #             scale = 0.9 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

        #             # get smpl parameters
        #             ## body resides in multi-ioi coordidate, where camera 0 is world zero.
        #             with open(smplx_param, 'rb') as f:
        #                 body_params = pkl.load(f)
        #                 # in ioi coordinates: cam 0
        #                 beta = body_params['betas']
        #                 pose_aa = body_params['body_pose']
        #                 pose_rotmat = p3dt.axis_angle_to_matrix(torch.FloatTensor(pose_aa.reshape(-1,3))).numpy()

        #                 transl = body_params['transl']
        #                 global_orient = body_params['global_orient']
        #                 global_orient = p3dt.axis_angle_to_matrix(torch.FloatTensor(global_orient.reshape(-1,3))).numpy()

        #             smpl_body_cam0 = body_model(betas=torch.FloatTensor(beta).to('cuda')) # canonical body with shape
        #             vertices_cam0 = smpl_body_cam0.vertices.detach().cpu().numpy().squeeze()
        #             joints_cam0 = smpl_body_cam0.joints.detach().cpu().numpy()
        #             pelvis_cam0 = joints_cam0[:, 0, :]


        #             ## rigid transformation between multi-ioi and Leica scan (world)
        #             with open(ioi2scan_fn, 'r') as f:
        #                 ioi2scan_dict = json.load(f)
        #                 R_ioi2world = np.array(ioi2scan_dict['R']) # Note: R is transposed
        #                 t_ioi2world= np.array(ioi2scan_dict['t']).reshape(1, 3)

        #             # get SMPL params in world coordinates
        #             # import ipdb; ipdb.set_trace()
        #             global_orient_world = np.matmul(R_ioi2world.T, global_orient)
        #             transl_world = np.matmul((pelvis_cam0+transl), R_ioi2world) + t_ioi2world - pelvis_cam0 # right multiplication to avoid transpose
        #             full_pose_rotmat_world = np.concatenate((global_orient_world, pose_rotmat), axis=0).squeeze()
        #             theta_world = batch_rot2aa(torch.FloatTensor(full_pose_rotmat_world)).reshape(-1, 66).cpu().numpy()

        #             smpl_body_world = body_model(betas=torch.FloatTensor(beta).to('cuda'),
        #                                 body_pose=torch.FloatTensor(theta_world[:, 3:]).to('cuda'),
        #                                 transl=torch.FloatTensor(transl_world).to('cuda'),
        #                                 global_orient=torch.FloatTensor(theta_world[:, :3]).to('cuda'))
        #             vertices_world = smpl_body_world.vertices.detach().cpu().numpy().squeeze()
        #             joints3d_world = smpl_body_world.joints[:, 25:, :].detach().cpu().numpy().squeeze()

        #             mesh = trimesh.Trimesh(vertices_world, body_model.faces,
        #                                    process=False,
        #                                    maintain_order=True)
        #             mesh.export('gt_mesh_world_smplx.obj')



        #             # smpl_body_world = body_model(betas=torch.FloatTensor(beta).to('cuda'),
        #             #                     body_pose=torch.FloatTensor(pose_rotmat[None, ...]).to('cuda'),
        #             #                     transl=torch.FloatTensor(transl_world[None, ...]).to('cuda'),
        #             #                     global_orient=torch.FloatTensor(global_orient_world[None, ...]).to('cuda'),
        #             #                     left_hand_pose=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 15, -1, -1).to('cuda'),
        #             #                     right_hand_pose=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 15, -1, -1).to('cuda'),
        #             #                     leye_pose= torch.eye(3).reshape(1, 1, 3, 3).expand(batch_size, -1, -1, -1),
        #             #
        #             #                       pose2rot=False)
        #             # vertices_world = smpl_body_world.vertices.detach().cpu().numpy().squeeze()
        #             # joints3d_world = smpl_body_world.joints[:, 25:, :].detach().cpu().numpy().squeeze()

        #             # mesh = trimesh.Trimesh(vertices_world, body_model.faces,
        #             #                        process=False,
        #             #                        maintain_order=True)
        #             # mesh.export('gt_mesh_world_smplx.obj')

        #             # get SMPL params in camera coordinates
        #             global_orient_cam = np.matmul(camR, global_orient)
        #             transl_cam = np.matmul(camR, (pelvis_cam0 + transl).T).T + camT - pelvis_cam0
        #             full_pose_rotmat_cam = np.concatenate((global_orient_cam, pose_rotmat), axis=0).squeeze()
        #             theta_cam = batch_rot2aa(torch.FloatTensor(full_pose_rotmat_cam)).reshape(-1, 66).cpu().numpy()
        #             # smpl_body_cam = body_model(betas=torch.FloatTensor(beta).to('cuda'),
        #             #                          body_pose=torch.FloatTensor(pose_rotmat).to('cuda'),
        #             #                          transl=torch.FloatTensor(transl_cam).to('cuda'),
        #             #                            global_orient=torch.FloatTensor(global_orient_cam).to('cuda'),
        #             #                            pose2rot=False)
        #             # vertices_cam = smpl_body_cam.vertices.detach().cpu().numpy().squeeze()
        #             # joints3d_cam = smpl_body_cam.joints[:, 25:, :].detach().cpu().numpy().squeeze()
        #             #
        #             # mesh = trimesh.Trimesh(vertices_cam, body_model.faces,
        #             #                        process=False,
        #             #                        maintain_order=True)
        #             # mesh.export('mesh_in_cam0.obj')

        #             # read GT 2D keypoints
        #             K = np.eye(3, dtype=np.float)
        #             K[0, 0] = focal_length_x / downsample_factor
        #             K[1, 1] = focal_length_y / downsample_factor
        #             K[:2, 2:] = camC.T / downsample_factor
        #             # projected_points = (K @ joints3d_cam.T).T
        #             # joints2d = projected_points[:, :2] / np.hstack((projected_points[:, 2:], projected_points[:, 2:]))
        #             # part = np.hstack((joints2d, np.ones((joints2d.shape[0], 1))))

        #             # get openpose 2D keypoints
        #             try:
        #                 with open(openpose_path, 'r') as f:
        #                     openpose = json.load(f)
        #                 openpose = np.array(openpose['people'][0]['pose_keypoints_2d']).reshape([-1, 3])
        #             except:
        #                 print(f'No openpose !! Missing {openpose_path}')
        #                 continue

        #             # get camera parameters wrt to scan
        #             R_worldtocam = np.matmul(camR, R_ioi2world) # Note: R_ioi2world is transposed
        #             T_worldtocam = -t_ioi2world + camT

        #             # ground offset
        #             # ground_offset = ground_eq[2]/

        #             # # get stability labels: 1: stable, 0: unstable but in contact, -1: unstable and not `in contact
        #             # in_bos_label, contact_label, contact_mask = vis_smpl_with_ground(theta_world, transl_world, beta, seq_i,
        #             #                                                      vis_path,
        #             #                                                      start_idx=frame_num,
        #             #                                                      sub_sample=1,
        #             #                                                      ground_offset=ground_offset,
        #             #                                                      smpl_batch_size=1,
        #             #                                                      visualize=False)
        #             # in_bos_label = in_bos_label.detach().cpu().numpy()
        #             # contact_label = contact_label.detach().cpu().numpy()
        #             # contact_mask = contact_mask.detach().cpu().numpy()

        #             # visualize world smpl on ground plane
        #             # if visualize:
        #             #     if cam_num == 0:
        #             #         vis_smpl_with_ground(theta_world, transl_world, beta, split+'_'+seq_i, vis_path,
        #             #                              start_idx=frame_num,
        #             #                              sub_sample=1,
        #             #                              ground_offset=ground_offset,
        #             #                              smpl_batch_size=1,
        #             #                              visualize=True)


        #             # ## visualize projected points
        #             # img = cv2.imread(img_path)
        #             # joints2d = joints2d.astype(np.int)
        #             # img[joints2d[:, 1], joints2d[:, 0], :] = [0, 255, 0]

        #             # read GT 3D pose in cam coordinates
        #             # S24 = joints3d_cam
        #             # pelvis_cam = (S24[[2], :] + S24[[3], :]) / 2
        #             # S24 -= pelvis_cam
        #             # S24 = np.hstack([S24, np.ones((S24.shape[0], 1))])

        #             # read GT 3D pose in world coordinates
        #             # S24_world = joints3d_world
        #             # S24_world = np.hstack([S24_world, np.ones((S24_world.shape[0], 1))])

        #             # store data
        #             imgnames_.append(img_path)
        #             centers_.append(center)
        #             scales_.append(scale)
        #             # parts_.append(part)
        #             # Ss_.append(S24)
        #             # Ss_world_.append(S24_world)
        #             openposes_.append(openpose)
        #             poses_.append(theta_cam.squeeze())
        #             transls_.append(transl.squeeze())
        #             poses_world_.append(theta_world.squeeze())
        #             transls_world_.append(transl_world.squeeze())
        #             shapes_.append(beta.squeeze())
        #             cams_r_.append(R_worldtocam.tolist())
        #             # Todo: note that T_worldtocam here is (1,3) whereas in h36m T_worldtocam is (1,3)
        #             cams_t_.append(T_worldtocam.tolist())
        #             cams_k_.append(K.tolist())
        #             # in_bos_label_.append(in_bos_label)
        #             # contact_label_.append(contact_label)
        #             ground_offset_.append(ground_offset)


        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, f'rich_{scene_name}_smplx.npz')
        np.savez(out_file, imgname=imgnames_,
                        #    center=centers_,
                        #    scale=scales_,
                        #    part=parts_,
                        #    S=Ss_,
                        #    S_world=Ss_world_,
                        pose=poses_,
                        transl=transls_,
                        shape=shapes_,
                        #    openpose=openposes_,
                        #    pose_world=poses_world_,
                        #    transl_world=transls_world_,
                        #    cam_r=cams_r_,
                        #    cam_t=cams_t_,
                        cam_k=cams_k_,
                        #    in_bos_label=in_bos_label_,
                        contact_label=contact_label_,
                        #    ground_offset=ground_offset_
                            scene_seg=scene_seg_,
                            part_seg=part_seg_
                )
        print('Saved to ', out_file)

def rectify_pose(camera_r, body_aa):
    body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa


def extract_cam_param_xml(xml_path: str = '', dtype=float):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = np.array([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)

    rotation = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]],
                         [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]],
                         [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = np.array([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [-extrinsics_mat[0] * extrinsics_mat[3] - extrinsics_mat[4] * extrinsics_mat[7] - extrinsics_mat[8] *
                  extrinsics_mat[11],
                  -extrinsics_mat[1] * extrinsics_mat[3] - extrinsics_mat[5] * extrinsics_mat[7] - extrinsics_mat[9] *
                  extrinsics_mat[11],
                  -extrinsics_mat[2] * extrinsics_mat[3] - extrinsics_mat[6] * extrinsics_mat[7] - extrinsics_mat[10] *
                  extrinsics_mat[11]]

    cam_center = np.array([cam_center], dtype=dtype)

    k1 = np.array([distortion_vec[0]], dtype=dtype)
    k2 = np.array([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2

# rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/work/achatterjee/rich/npzs', split='train')
# rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/fast/achatterjee/rich/scene_npzs', split='val')
# rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/fast/achatterjee/rich/scene_npzs/test', split='test')
# rich_extract(dataset_path='/ps/scratch/ps_shared/stripathi/4yogi/RICH/val/', out_path='/home/achatterjee/rich_ext', split='val')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    args = parser.parse_args()

    rich_extract(img_dataset_path='/is/cluster/work/achatterjee/rich', out_path='/is/cluster/fast/achatterjee/rich/scene_npzs/test', scene_indx=args.index, split='test')