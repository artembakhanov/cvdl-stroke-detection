import click
import os
import re
from os.path import isfile, join
from pathlib import Path
import shutil
import random
from PIL import Image

from atlas_parser.nii_to_sequence import AtlasParser


class AtlasReformer:
    def __init__(self, dataset_path: str, new_dataset_path: str):
        self.dataset_path = dataset_path
        self.new_dataset_path = new_dataset_path

    def _get_brain_and_masks(self, site):
        self.brain_masks: list[dict[str, str]] = []
        self.stroke_sequences = []
        self.no_stroke_sequences = []
        samples = os.listdir(site)
        for sample in samples:
            patient_files = os.listdir(os.path.join(site, sample, 't01'))
            patient_files = [os.path.join(site, sample, 't01', pf) for pf in patient_files]
            masks = list(filter(lambda f: 'LesionSmooth' in f, patient_files))
            brain = list(filter(lambda f: 'LesionSmooth' not in f, patient_files))[0]
            self.brain_masks.append({'brain': brain, 'masks': masks})

    def _create_new_dataset_path(self):
        for ar in ['train', 'test']:
            Path(f'{self.new_dataset_path}/{ar}').mkdir(parents=True, exist_ok=True)
            for arr in ['stroke', 'no_stroke']:
                Path(f'{self.new_dataset_path}/{ar}/{arr}').mkdir(parents=True, exist_ok=True)
    
    def _train_test_split(self, p=0.8):
        train_len = int(len(self.stroke_sequences) * p)
        print(f'*train: {train_len} samples\n*test: {len(self.stroke_sequences) - train_len} samples')
        self.sequences = {
            'train': {
                'stroke': self.stroke_sequences[:train_len],
                'no_stroke': self.no_stroke_sequences[:train_len]
            },
            'test': {
                'stroke': self.stroke_sequences[train_len:],
                'no_stroke': self.no_stroke_sequences[train_len:]
            }
        }

    def _parse_patients(self):
        for patient in self.brain_masks:
            brain_path, masks_paths = patient['brain'], patient['masks']
            try:
                atlas_parser = AtlasParser(brain_path, masks_paths, sequence_len=5)
                stroke_sequence, no_stroke_sequence = atlas_parser.nii_to_sequences()
                self.stroke_sequences.append(stroke_sequence)
                self.no_stroke_sequences.append(no_stroke_sequence)
            except:
                print('the stroke part is too small:', brain_path, masks_paths)


    def _save(self):
        for split in self.sequences:
            for category in self.sequences[split]:
                for i, sequence in enumerate(self.sequences[split][category]):
                    save_to = f'{self.new_dataset_path}/{split}/{category}/{i+1}'
                    Path(save_to).mkdir(parents=True, exist_ok=True)
                    for j, img in enumerate(sequence):
                        img.save(f'{save_to}/{j+1}.png')

    def run(self):
        sites = [os.path.join(self.dataset_path, f'Site{i}') for i in range(1, 10)]
        for i, site in enumerate(sites):
            self._get_brain_and_masks(site)
            for _ in range(4):
                self._parse_patients()
            self._train_test_split()
            self._create_new_dataset_path()
            self._save()
            print(f'> finished with Site{i}!')


@click.command()
@click.option('--dataset-path', default='/home/whoorma/Documents/neusomething/ATLAS_R1.1', prompt='path to the dataset', type=str)
@click.option('--new-dataset-path', default='/home/whoorma/Documents/neusomething/ATLAS_REFORMED', prompt='path to the new dataset', type=str)
def main(dataset_path: str, new_dataset_path: str):
    try:
        shutil.rmtree(new_dataset_path)
    except:
        pass
    atlas_reformer = AtlasReformer(dataset_path, new_dataset_path)
    atlas_reformer.run()
    print(f'go to {new_dataset_path} and check it out :)')

if __name__ == '__main__':
    main()