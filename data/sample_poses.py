"""Sample poses from AMASS dataset for training
This is based(almost copied) on data preparation of VPoser training"""


import os
import numpy as np


def prepare_vposer_datasets(vposer_dataset_dir, amass_splits, amass_dir, mode='train'):
    keep_rate = 0.3
    amass_datas = amass_splits[mode]
    for amass_data in amass_datas:
        ds_dir = os.path.join(amass_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))
        # pose_body = []
        # root_orient = []
        # betas_all = []
        # motion_frames = []
        if not os.path.exists(os.path.join(vposer_dataset_dir, amass_data)):
            os.makedirs(os.path.join(vposer_dataset_dir, amass_data))
        for seq in seqs:
            pose_body = []
            root_orient = []
            betas_all = []
            motion_frames = []
            translation = []
            if 'LICENSE' in seq:
                continue
            out_path = os.path.join(vposer_dataset_dir, amass_data, seq + '.npz')
            if os.path.exists(out_path):
                continue
            npz_fnames = sorted(os.listdir(os.path.join(ds_dir, seq)))
            for npz_fname in npz_fnames:
                if 'female' in npz_fname or 'male' in npz_fname or 'neutral' in npz_fname or 'shape' in npz_fname:
                    continue
                cdata = np.load(os.path.join(ds_dir, seq,npz_fname))
                print(os.path.join(ds_dir, seq,npz_fname))
                N = len(cdata['poses'])

                # skip first and last frames to avoid initial standard poses, e.g. T pose
                # cdata_ids = np.random.choice(list(range(int(0.1 * N), int(0.9 * N), 1)), int(keep_rate * 0.8 * N),
                #                              replace=False)
                cdata_ids = list(range(0, N))
                motion_frames.append(len(cdata_ids))
                if len(cdata_ids) < 1:
                    continue
                # print(N, len(cdata_ids))
                fullpose = cdata['poses'][:].astype(np.float32) # cdata['poses']:(524,156)
                betas = cdata['betas'].astype(np.float32) # accd 16
                # betas = np.expand_dims(betas, 0)
                # betas = np.repeat(betas, len(cdata_ids), 0)
                pose_body.extend(fullpose[:, :66])
                translation.extend(cdata['trans'])
                
                # betas_all.extend(betas)
                root_orient.extend(fullpose[:, :3])
                # ipdb.set_trace()

            np.savez(out_path, pose_body=np.array(pose_body), root_orient= np.array(root_orient), betas=np.array(betas), motion_frames=np.array(motion_frames), translation=np.array(translation))
            print(mode, amass_data, seq, len(root_orient))



if __name__ == "__main__":

    # from data.data_splits import amass_splits
    amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje' ,'BMLhandball' ,
              'BMLmovi', 'EKUT', 'TCD_handMocap',
              'ACCAD']
    }

    posendf_data_dir = '/media/xky/_data/posendf/amass2humandiffusion_data'
    amass_dir = '/media/xky/_data/posendf/amass'

    prepare_vposer_datasets(posendf_data_dir, amass_splits, amass_dir, mode='train')