# This file comes from CubiCasa5K, https://github.com/CubiCasa/CubiCasa5k (floortrans/loaders/augmentations.py, class RotateNTurns).
# Copyright (c) 2019 CubiCasa. Licensed under the Creative Commons
# Attribution-NonCommercial 4.0 International License, see LICENSE in this folder.
# Changes for ChatChTG: only RotateNTurns is kept. In the half turn, I junction 3
# goes to channel 1. The original wrote it to channel 4, which the L junctions overwrite.

import torch


class RotateNTurns(object):

    def rot_tensor(self, t, n):
        # One turn clock wise
        if n == 1:
            t = t.flip(2).transpose(3, 2)
        # One turn counter clock wise
        elif n == -1:
            t = t.transpose(3, 2).flip(2)
        # Two turns clock wise
        elif n == 2:
            t = t.flip(2).flip(3)

        return t

    def rot_points(self, t, n):
        # Swapping corner ts 
        t_sorted = t.clone().detach()
        # One turn clock wise
        if n == 1:
            # I junctions
            t_sorted[:, 1] = t[:, 0]
            t_sorted[:, 2] = t[:, 1]
            t_sorted[:, 3] = t[:, 2]
            t_sorted[:, 0] = t[:, 3]
            # L junctions
            t_sorted[:, 5] = t[:, 4]
            t_sorted[:, 6] = t[:, 5]
            t_sorted[:, 7] = t[:, 6]
            t_sorted[:, 4] = t[:, 7]
            # T junctions
            t_sorted[:, 9] = t[:, 8]
            t_sorted[:, 10] = t[:, 9]
            t_sorted[:, 11] = t[:, 10]
            t_sorted[:, 8] = t[:, 11]
            # Opening corners
            t_sorted[:, 15] = t[:, 13]
            t_sorted[:, 16] = t[:, 14]
            t_sorted[:, 14] = t[:, 15]
            t_sorted[:, 13] = t[:, 16]
            # Icon corners
            t_sorted[:, 18] = t[:, 17]
            t_sorted[:, 20] = t[:, 18]
            t_sorted[:, 17] = t[:, 19]
            t_sorted[:, 19] = t[:, 20]
        # One turn counter clock wise
        elif n == -1:
            # I junctions
            t_sorted[:, 3] = t[:, 0]
            t_sorted[:, 0] = t[:, 1]
            t_sorted[:, 1] = t[:, 2]
            t_sorted[:, 2] = t[:, 3]
            # L junctions
            t_sorted[:, 7] = t[:, 4]
            t_sorted[:, 4] = t[:, 5]
            t_sorted[:, 5] = t[:, 6]
            t_sorted[:, 6] = t[:, 7]
            # T junctions
            t_sorted[:, 11] = t[:, 8]
            t_sorted[:, 8] = t[:, 9]
            t_sorted[:, 9] = t[:, 10]
            t_sorted[:, 10] = t[:, 11]
            # Opening corners
            t_sorted[:, 16] = t[:, 13]
            t_sorted[:, 15] = t[:, 14]
            t_sorted[:, 13] = t[:, 15]
            t_sorted[:, 14] = t[:, 16]
            # Icon corners
            t_sorted[:, 19] = t[:, 17]
            t_sorted[:, 17] = t[:, 18]
            t_sorted[:, 20] = t[:, 19]
            t_sorted[:, 18] = t[:, 20]
        # Two turns clock wise
        elif n == 2:
            t_sorted = t.clone().detach()
            # I junctions
            t_sorted[:, 2] = t[:, 0]
            t_sorted[:, 3] = t[:, 1]
            t_sorted[:, 0] = t[:, 2]
            t_sorted[:, 1] = t[:, 3]
            # L junctions
            t_sorted[:, 6] = t[:, 4]
            t_sorted[:, 7] = t[:, 5]
            t_sorted[:, 4] = t[:, 6]
            t_sorted[:, 5] = t[:, 7]
            # T junctions
            t_sorted[:, 10] = t[:, 8]
            t_sorted[:, 11] = t[:, 9]
            t_sorted[:, 8] = t[:, 10]
            t_sorted[:, 9] = t[:, 11]
            # Opening corners
            t_sorted[:, 14] = t[:, 13]
            t_sorted[:, 13] = t[:, 14]
            t_sorted[:, 16] = t[:, 15]
            t_sorted[:, 15] = t[:, 16]
            # Icon corners
            t_sorted[:, 20] = t[:, 17]
            t_sorted[:, 19] = t[:, 18]
            t_sorted[:, 18] = t[:, 19]
            t_sorted[:, 17] = t[:, 20]
        elif n == 0:
            return t_sorted

        return t_sorted

    def __call__(self, sample, data_type, n):
        if data_type == 'tensor':
            return self.rot_tensor(sample, n)
        elif data_type == 'points':
            return self.rot_points(sample, n)

