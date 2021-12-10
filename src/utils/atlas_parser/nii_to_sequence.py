import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import random
from pathlib import Path
from enum import Enum
from typing import Optional


class AtlasParser:
    def __init__(self,
                 brain_path: str,
                 masks_paths: list[str],
                 sequence_len: int,
                 print_info: bool = False,
                 save_to: Optional[dict[str, str]] = None):
        self.brain_path = brain_path
        self.masks_paths = masks_paths
        self.sequence_len = sequence_len
        self.print_info = print_info
        self.save_to = save_to

    def _get_chunks(self, slices, ret_message=True):
        chunks = []
        start, finish = None, None
        for elem in slices:
            if start is None:
                start = elem
            elif elem - finish != 1:
                chunks.append((start, finish))
                start = elem
            finish = elem
        chunks.append((start, finish))
        if ret_message:
            return ''.join(f'| {mi}-{ma} |' for mi, ma in chunks)
        return chunks

    def _get_sequence(self, brain, is_stroke: bool):
        '''
        brain: 3D array of initial model
        save_to: path to folder
        sequence_length: length of sequences to be extracted from the model
        '''
        indices = sorted(random.sample(range(brain.shape[0]), self.sequence_len))
        sequence = []
        if self.save_to:
            save_to = self.save_to['stroke' if is_stroke else 'no_stroke']
            Path(save_to).mkdir(parents=True, exist_ok=True)
        for i in indices:
            cur = brain[i, :, :]
            cur = cur.astype(np.uint8)
            cur = Image.fromarray(cur)
            if self.save_to:
                cur.save(f'{save_to}/{i}.png')
            sequence.append(cur)
        return sequence

    def _get_no_stroke(self, brain, mask):
        # invert mask: get 0s where there is a stroke
        mask_inv = mask.max() - mask
        # get mask: for each slice without brain stroke there will be True
        mask_no_stroke = np.array([mask_inv[i, :, :].all() for i in range(mask_inv.shape[0])])
        if self.print_info:
            trues = (mask_no_stroke == True).nonzero()[0] # take indices
            print(f'slices without brain_stroke: {self._get_chunks(trues)}')
        # will contain zeros at planes without stroke
        brain_no_stroke = (mask_no_stroke * brain.transpose()).transpose()
        # remove planes with only zeros (these are either useless or contained stroke)
        brain_no_stroke = np.array([brain_no_stroke[i, :, :] for i in range(brain_no_stroke.shape[0]) if brain_no_stroke[i, :, :].any()])
        return brain_no_stroke

    def _get_stroke(self, brain, mask):
        # get mask: for each slice with brain stroke there will be True
        mask_stroke = np.array([mask[i, :, :].any() for i in range(mask.shape[0])])
        if self.print_info:
            trues = (mask_stroke == True).nonzero()[0] # take indices
            print(f'slices with brain stroke: {self._get_chunks(trues)}')
        # will contain ones at planes with stroke
        brain_stroke = (mask_stroke * brain.transpose()).transpose()
        # remove planes with only zeros (these are either useless or contained no stroke)
        brain_stroke = np.array([brain_stroke[i, :, :] for i in range(brain_stroke.shape[0]) if brain_stroke[i, :, :].any()])
        return brain_stroke

    def _get_samples(self, brain, masks: list):
        # merge all masks into one, it will contain 255.0 at places with stroke
        mask = sum(masks)
        # do not take the upper part of a brain model
        min_idx, max_idx = int(mask.shape[0] * 0.1), int(mask.shape[0] * 0.7)
        # we need the 0th dimension (view from above)
        brain = brain[min_idx:max_idx, :, :]
        mask = mask[min_idx:max_idx, :, :]
        # get two 3D parts: with and without stroke
        stroke_part = self._get_stroke(brain, mask)
        no_stroke_part = self._get_no_stroke(brain, mask)
        stroke_sequence = self._get_sequence(stroke_part, is_stroke=True)
        no_stroke_sequence = self._get_sequence(no_stroke_part, is_stroke=False)
        return stroke_sequence, no_stroke_sequence

    def _read_nii(self, path: str):
        img = sitk.ReadImage(path)
        # return 3D numpy array
        return sitk.GetArrayFromImage(img)


    def nii_to_sequences(self):
        # read brain
        brain = self._read_nii(self.brain_path)
        # read all masks
        masks = list(map(self._read_nii, self.masks_paths))
        for mask in masks:
            assert mask.shape == brain.shape
        # return one sequence with stroke and one sequence without stroke
        return self._get_samples(brain, masks)


if __name__ == '__main__':
    '''
    example of how you can use AtlasParser
    '''
    # make sure 'test_nii_to_seq' folder is in your current working directory
    brain_path = os.path.join(os.getcwd(), 'test_nii_to_seq/031769_t1w_deface_stx.nii.gz')
    masks_paths = [
        os.path.join(os.getcwd(), 'test_nii_to_seq/031769_LesionSmooth_stx.nii.gz'),
        os.path.join(os.getcwd(), 'test_nii_to_seq/031769_LesionSmooth_1_stx.nii.gz')
    ]
    sequence_len = 15
    print_info = True
    save_to = {
        'stroke': os.path.join(os.getcwd(), 'stroke'),
        'no_stroke': os.path.join(os.getcwd(), 'no_stroke')
    }
    atlas_parser = AtlasParser(brain_path, masks_paths, sequence_len, print_info, save_to)
    stroke_sequence, no_stroke_sequence = atlas_parser.nii_to_sequences()
