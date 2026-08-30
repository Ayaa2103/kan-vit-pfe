# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels. Originally backed by the cumm/spconv sparse
conv library's Point2VoxelCPU3d generator; replaced with a pure-NumPy
implementation (see preprocess() below) because that native extension
(built for the NumPy 1.x C-ABI) hard-crashes the process when fed NumPy
2.x arrays on this platform. No cumm/spconv import is needed anymore.
"""
import sys

import numpy as np
import torch
from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def preprocess(self, pcd_np):
        """
        Pure-NumPy voxelization, used in place of cumm/spconv's compiled
        Point2VoxelCPU3d generator. That native extension (built for the
        NumPy 1.x C-ABI) hard-crashes the process when fed NumPy 2.x arrays
        on this platform, so we reimplement the same binning/padding
        contract here. Output matches spconv's convention: voxel_coords
        columns are ordered [z, y, x] (see pillar_vfe.py / point_pillar_scatter.py).
        """
        voxel_size = np.asarray(self.voxel_size, dtype=np.float64)
        pc_range = np.asarray(self.lidar_range, dtype=np.float64)
        nx, ny, nz = (int(v) for v in self.grid_size)

        pts = np.asarray(pcd_np, dtype=np.float32)
        in_range = np.all((pts[:, :3] >= pc_range[:3]) &
                          (pts[:, :3] < pc_range[3:]), axis=1)
        pts = pts[in_range]

        C = pts.shape[1]
        if pts.shape[0] == 0:
            return {
                'voxel_features': np.zeros((0, self.max_points_per_voxel, C), dtype=np.float32),
                'voxel_coords': np.zeros((0, 3), dtype=np.int32),
                'voxel_num_points': np.zeros((0,), dtype=np.int32),
            }

        idx_xyz = np.floor((pts[:, :3] - pc_range[:3]) / voxel_size).astype(np.int64)
        idx_xyz[:, 0] = np.clip(idx_xyz[:, 0], 0, nx - 1)
        idx_xyz[:, 1] = np.clip(idx_xyz[:, 1], 0, ny - 1)
        idx_xyz[:, 2] = np.clip(idx_xyz[:, 2], 0, nz - 1)

        voxel_key = (idx_xyz[:, 2] * ny + idx_xyz[:, 1]) * nx + idx_xyz[:, 0]

        unique_keys, first_idx, inverse, counts = np.unique(
            voxel_key, return_index=True, return_inverse=True, return_counts=True)

        if len(unique_keys) > self.max_voxels:
            order = np.argsort(first_idx)[:self.max_voxels]
            keep_keys = unique_keys[order]
            keep_mask = np.isin(voxel_key, keep_keys)
            pts = pts[keep_mask]
            idx_xyz = idx_xyz[keep_mask]
            voxel_key = voxel_key[keep_mask]
            unique_keys, first_idx, inverse, counts = np.unique(
                voxel_key, return_index=True, return_inverse=True, return_counts=True)

        num_voxels = len(unique_keys)
        voxel_features = np.zeros((num_voxels, self.max_points_per_voxel, C), dtype=np.float32)
        voxel_num_points = np.minimum(counts, self.max_points_per_voxel).astype(np.int32)
        voxel_coords = np.zeros((num_voxels, 3), dtype=np.int32)
        voxel_coords[:, 0] = idx_xyz[first_idx, 2]  # z
        voxel_coords[:, 1] = idx_xyz[first_idx, 1]  # y
        voxel_coords[:, 2] = idx_xyz[first_idx, 0]  # x

        order = np.argsort(inverse, kind='stable')
        sorted_inverse = inverse[order]
        group_start = np.searchsorted(sorted_inverse, np.arange(num_voxels))
        slot_in_group = np.arange(len(sorted_inverse)) - group_start[sorted_inverse]

        valid = slot_in_group < self.max_points_per_voxel
        voxel_features[sorted_inverse[valid], slot_in_group[valid]] = pts[order][valid]

        data_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
        }

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}
