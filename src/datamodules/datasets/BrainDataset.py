import os
import random
import shutil
import glob
from typing import Callable, Optional
from urllib.error import URLError

import cv2
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
from torchvision.transforms.functional import rotate
from tqdm import tqdm

from src.datamodules.datasets.BaseDataset import BaseDataset
from src.utils import utils

log = utils.get_logger(__name__)


class BrainDataset(BaseDataset):
    file_name = "Brain_Data_reordered2.zip"
    file_id = "10lbK8GkXVXVhW-tUJ55IVLpbVs4fJlDY"
    classes = {'Normal': 0, 'Stroke': 1}

    SEQUENCE_SIZE = 15

    def __init__(self,
                 root: str,
                 static_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 orphan: bool = False,
                 image_size: int = 128,
                 debug: bool = False,
                 dynamic_transform: Optional[Callable] = None,
                 rotate: bool = True,
                 rot_seq_prob: float = 0.4,
                 test: bool = False,
                 sequence_size: int = 15
                 ):
        super(BrainDataset, self).__init__(root, dynamic_transform, target_transform, download, orphan, image_size,
                                           debug)
        if orphan:
            return
        self.rotate = rotate
        self.rot_seq_prob = rot_seq_prob
        self.test = test
        self.sequence_size = sequence_size

        self.images_sequences = {}
        self.idx_to_img_num = {}
        self.min_seq_size = sequence_size // 2
        self.rot_seq_prob = rot_seq_prob
        self.distributions = []
        self.rotate = rotate

        log.info(f"Reading {'test' if test else 'train'} images")
        for category in self.classes.keys():  # Normal, Stroke
            new_dir = os.path.join(self.root, "Test" if test else "Train", category)
            sequences = self._get_sequences(new_dir)
            for i, sequence in tqdm(enumerate(sequences), desc=f"Reading {category} images", total=len(sequences)):
                key = (i, self.classes[category])
                self.idx_to_img_num[len(self.idx_to_img_num)] = key
                for record in sequence:
                    img = cv2.imread(f'{new_dir}/{record["filename"]}', 0)
                    if key in self.images_sequences.keys():
                        self.images_sequences[key].append({
                            'image': static_transform(image=img)['image'] if static_transform else img,
                            'order': record['order'],
                        })
                    else:
                        self.images_sequences[key] = [{
                            'image': static_transform(image=img)['image'] if static_transform else img,
                            'order': record['order'],
                        }]

    def _sample_from_sequence(self, sequence, max_step=1):
        # max_step == 2 => = average max step: can be: [1, 2, 3, 4], can be [1, 2, 5, 6] from (1, .., 8)
        if len(sequence) < self.min_seq_size:
            return None
        if len(sequence) == self.sequence_size:
            return [sequence]
        original_indices = list(map(lambda rec: int(rec['order']), sequence))
        if len(sequence) < self.sequence_size:
            sampled_sequences = []
            for i in range(self.sequence_size // max_step):
                # repeat random ones
                indices_to_repeat = sorted(random.choices(original_indices, k=self.sequence_size))
                new_sequence = []
                for idx in indices_to_repeat:
                    for rec in sequence:
                        if int(rec['order']) == idx:
                            new_sequence.append(rec)
                sampled_sequences.append(new_sequence)
            return sampled_sequences
        # sample from sequence - consecutive elements or with max_step
        sampled_sequences = []
        for i in range(len(original_indices) - self.sequence_size):
            indices = None
            for j in range(i + self.sequence_size * max_step, i, -1):
                if indices is None:
                    try:
                        indices = original_indices[i:j]
                    except:
                        pass
            # sample random `self.sequence_size` elements from `elements`
            indices_to_keep = sorted(random.sample(indices, k=self.sequence_size))
            new_sequence = []
            for idx in indices_to_keep:
                for rec in sequence:
                    if int(rec['order']) == idx:
                        new_sequence.append(rec)
            sampled_sequences.append(new_sequence)
        return sampled_sequences

    def _reconstruct_filename(self, num, order):
        return f'{num} ({order}).jpg'

    def _get_sequences(self, path):
        sequences = {}
        for file in os.listdir(path):
            if '.jpg' not in file:
                continue
            # file name: 50 (49).jpg -> num: 50, order: 49
            num, order = file.split(' ')
            order = order.split('.')[0][1:-1]
            record = {'order': order, 'filename': file}
            if num in sequences:
                sequences[num].append(record)
            else:
                sequences[num] = [record]
        # sequences = dict(filter(lambda kv: len(kv[1]) > 4, sequences.items()))
        sequences = dict(map(lambda kv: (kv[0], sorted(kv[1], key=lambda x: int(x['order']))), sequences.items()))
        short_sequences = []
        for key, seq in sequences.items():
            sampled_sequences = self._sample_from_sequence(seq)
            if sampled_sequences:
                short_sequences.append(sampled_sequences)
        # flatten
        short_sequences = [seq for sampled_seqs in short_sequences for seq in sampled_seqs]
        return short_sequences

    def _rotate_tensor_sequence(self, sequence):
        if random.uniform(0, 1) <= self.rot_seq_prob:
            angle = random.randint(-180, 180)
            sequence = map(lambda img: rotate(img, angle), sequence)
            return list(sequence)
        return sequence

    def __len__(self):
        return len(self.idx_to_img_num)

    def __getitem__(self, idx):
        key = self.idx_to_img_num[idx]
        sequence = list(
            map(lambda rec: self.transform(image=rec['image'])['image'] if self.transform else rec['image'],
                self.images_sequences[key]))
        if self.rotate:
            sequence = self._rotate_tensor_sequence(sequence)
        return sequence, key[1]

    def _download(self) -> None:
        """Download the brain strokes data if it doesn't exist already."""

        if self._check_exists():
            log.info(f"{self.__class__.__name__} is already downloaded. No need to download")
            return

        # download files
        try:
            log.info("Downloading {}...".format(self.file_name))
            download_file_from_google_drive(self.file_id, self._raw_folder, filename=self.file_name)
            log.info("Extracting {}...".format(self.file_name))
            extract_archive(os.path.join(self._raw_folder, self.file_name), self.root)

            files = glob.glob(f"{self.root}/Brain_Data_reordered2/*")
            for f in files:
                shutil.move(f, self.root)

            shutil.rmtree(f"{self.root}/Brain_Data_reordered2/")
            self._create_meta()
        except URLError as error:
            log.error(
                "Failed to download (trying next):\n{}".format(error)
            )
        finally:
            pass
