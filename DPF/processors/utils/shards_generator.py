import os
import tarfile
import csv
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from DPF.utils.utils import get_file_extension


def is_list_has_no_duplicates(array):
    return len(array) == len(set(array))


class ShardsGenerator:
    def __init__(self, df, save_path, processes=8,
                 images_per_tar=1000, force=False, rename_images=False,
                 save_csv=True, imagename_column="image_name", columns_to_add=None):
        if columns_to_add is None:
            columns_to_add = []
        self.df = df
        self.save_path = save_path
        self.processes = processes
        self.images_per_tar = images_per_tar
        self.force = force
        self.rename_images = rename_images
        self.save_csv = save_csv
        self.imagename_column = imagename_column
        self.columns_to_add = columns_to_add

    def _flush_chunk(self, samples, shard_path, shard_number):
        tar_path = shard_path + '.tar'
        csv_path = shard_path + '.csv'

        tar = tarfile.open(tar_path, "w")
        if self.save_csv:
            csvfile = open(csv_path, 'w')
            writer = csv.writer(csvfile)
            writer.writerow([self.imagename_column]+self.columns_to_add)

        for c, data in enumerate(samples):
            image_path = data[0]
            if self.rename_images:
                image_name = str(self.images_per_tar*shard_number+c) \
                             + get_file_extension(image_path)
            else:
                image_name = os.path.basename(image_path)
            if self.save_csv:
                writer.writerow([image_name, *data[1:]])
            tar.add(image_path, arcname=image_name)

        tar.close()
        if self.save_csv:
            csvfile.close()

    def _mp_flush_chunk(self, params):
        return self._flush_chunk(*params)

    def run(self):
        save_path = self.save_path.rstrip('/')
        os.makedirs(self.save_path, exist_ok=True)
        assert self.force or len(os.listdir(self.save_path)) == 0, \
            "Directory is not empty. Set force=True to ignore this"
        assert set(self.columns_to_add).issubset(set(self.df.columns))

        all_columns = ['image_path']+self.columns_to_add
        params = []
        total = 0
        for chunk_id, (a, b) in tqdm(enumerate(zip(
            np.arange(0, self.df.shape[0], self.images_per_tar),
            np.arange(self.images_per_tar,
                      self.df.shape[0] + self.images_per_tar, self.images_per_tar),
        ))):
            chunk = self.df[a:b]
            shard_path = f'{self.save_path}/{chunk_id}'
            params.append(
                (zip(*[chunk[col] for col in all_columns]), shard_path, chunk_id)
            )
            image_names = chunk['image_name'].values
            assert is_list_has_no_duplicates(image_names), \
                "Image names are duplicated, set rename_images=True"

            total += len(chunk)

        process_map(self._mp_flush_chunk, params, max_workers=self.processes, chunksize=1)
