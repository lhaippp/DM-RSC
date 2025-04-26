import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path

class Dataset_fastec_ODM(Dataset):
    def __init__(self, root_dir, seq_len=3):
        self.seq_len = seq_len
        self.samples = self._build_samples(root_dir)
        self.transform = T.Resize((96, 128), antialias=True)
    def map_to_result_path(self, old_path_str, anchor="Fastec"):
        old_path = Path(old_path_str)
        parts = old_path.parts

        # 找到 anchor ("Fastec") 出现的位置
        try:
            idx = parts.index(anchor)
        except ValueError:
            raise ValueError(f"Given path does not contain the expected anchor: {anchor}")

        # 从 Fastec/ 后面开始截取
        subpath = Path(*parts[idx:])

        new_root = Path("/data/DM-RSC/result/result_fastec_MDM/sample-final")
        new_path = new_root / subpath / "flow_npy"
        return new_path
    
    def _build_samples(self, root_dir):
        samples = []
        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            if '190206' not in seq_path:
                continue

            num_frames = len(fnames) // 3 - 2
            for i in range(num_frames):
                #base_intra = seq_path.replace('Fastec', 'Fastec_intra_gma_m')
                base_intra = self.map_to_result_path(seq_path)
                base_inter = seq_path.replace('Fastec', 'Fastec_inter_gma')

                paths = {
                    'rs1': os.path.join(seq_path, f'{i:03d}_rolling.png'),
                    'rs2': os.path.join(seq_path, f'{i+1:03d}_rolling.png'),
                    'rs3': os.path.join(seq_path, f'{i+2:03d}_rolling.png'),
                    'gs2': os.path.join(seq_path, f'{i+1:03d}_global_middle.png'),
                    'flow_intra': os.path.join(base_intra, f'rs{i+1}_to_gs{i+1}.npy'),
                    'flow_intra1': os.path.join(base_intra, f'gs{i+1}_to_rs{i+1}.npy'),
                    'flow12': os.path.join(base_inter, f'frame{i}_to_{i+1}.npy'),
                    'flow32': os.path.join(base_inter, f'frame{i+2}_to_{i+1}.npy'),
                    'flow21': os.path.join(base_inter, f'frame{i+1}_to_{i}.npy'),
                    'flow23': os.path.join(base_inter, f'frame{i+1}_to_{i+2}.npy')
                }

                if not os.path.exists(paths['rs1']):
                    continue

                samples.append(paths)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]

        def load_image(path):
            img = cv2.imread(path)[:, :, [2, 1, 0]] / 255.0
            return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        def load_flow(path):
            return torch.from_numpy(np.load(path))

        rs1 = load_image(paths['rs1'])
        rs2 = load_image(paths['rs2'])
        rs3 = load_image(paths['rs3'])
        gs2 = load_image(paths['gs2'])

        flow = load_flow(paths['flow_intra'])
        flow1 = load_flow(paths['flow_intra1'])
        flow12 = load_flow(paths['flow12'])
        flow32 = load_flow(paths['flow32'])
        flow21 = load_flow(paths['flow21'])
        flow23 = load_flow(paths['flow23'])

        return {
            'rs1': rs1,
            'rs2': rs2,
            'rs3': rs3,
            'gs2': gs2,
            'flow': flow,
            'flow1': flow1,
            'flow12': flow12,
            'flow32': flow32,
            'flow21': flow21,
            'flow23': flow23,
            'path': paths
        }
