import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T


class Dataset_fastec_MDM(Dataset):
    def __init__(self, root_dir, seq_len=3):
        self.seq_len = seq_len
        self.samples = self._build_samples(root_dir)
        self.transform = T.Resize((96, 128), antialias=True)

    def _build_samples(self, root_dir):
        samples = []
        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            if '190206' not in seq_path:
                continue

            num_frames = len(fnames) // 3 - 2
            for i in range(num_frames):
                base_intra = seq_path.replace('Fastec', 'Fastec_intra_gma_m')
                base_inter = seq_path.replace('Fastec', 'Fastec_inter_gma')

                flow_intra = os.path.join(base_intra, f'rs{i+1}_to_gs{i+1}.npy')
                flow_intra1 = os.path.join(base_intra, f'gs{i+1}_to_rs{i+1}.npy')
                flow12 = os.path.join(base_inter, f'frame{i}_to_{i+1}.npy')
                flow23 = os.path.join(base_inter, f'frame{i+1}_to_{i+2}.npy')

                if not os.path.exists(flow12):
                    continue

                rs_path = os.path.join(seq_path, f'{i+1:03d}_rolling.png')
                gs_path = os.path.join(seq_path, f'{i+1:03d}_global_middle.png')

                samples.append([flow_intra, flow_intra1, flow12, flow23, rs_path, gs_path])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow_intra_path, flow_intra1_path, flow12_path, flow23_path, rs2_path, gs2_path = self.samples[idx]

        # Load flows
        intra_flow = np.load(flow_intra_path)
        intra_flow1 = np.load(flow_intra1_path)
        flow12 = np.load(flow12_path)
        flow23 = np.load(flow23_path)

        # Load images
        rs2 = cv2.imread(rs2_path)[:, :, [2, 1, 0]] / 255.0  # BGR to RGB
        gs2 = cv2.imread(gs2_path)[:, :, [2, 1, 0]] / 255.0

        rs2 = torch.tensor(rs2, dtype=torch.float32).permute(2, 0, 1)
        gs2 = torch.tensor(gs2, dtype=torch.float32).permute(2, 0, 1)

        return {
            'intra_flow': intra_flow,
            'intra_flow1': intra_flow1,
            'flow12': flow12,
            'flow23': flow23,
            'rs2_ori': rs2,
            'gs2_ori': gs2,
            'path': self.samples[idx]
        }
