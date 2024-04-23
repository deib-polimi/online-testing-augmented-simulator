import pathlib

import numpy as np
import pandas as pd
import pyiqa
import torch
from PIL import Image
from tqdm.contrib.concurrent import process_map

from utils.conf import DEFAULT_DEVICE
from utils.path_utils import get_images_from_folder, get_result_folders


# def run_on_folder(folder: pathlib.Path):
if __name__ == '__main__':

    for folder in [
        pathlib.Path("/media/banana/data/results/online-testing/online/stable_diffusion_inpainting/A-street-in-italy-photo-taken-from-a-car/before"),
        pathlib.Path("/media/banana/data/results/online-testing/online/stable_diffusion_inpainting/A-street-in-italy-photo-taken-from-a-car/after"),
    ]:
        # folder = pathlib.Path("/media/banana/data/results/online-testing/online/stable_diffusion_inpainting/A-street-in-netherlands-photo-taken-from-a-car/before/")

        # 0. Run configuration
        input_file = folder.joinpath("nr_iqa.csv")
        # if output_file.exists():
        #     return
        print(folder)

        df = pd.read_csv(input_file).fillna(0)

        print(df.mean(numeric_only=True))
        print(df.corr(numeric_only=True))



    # # 1. Define set of metrics
    # metrics = {
    #     'brisque': pyiqa.create_metric('brisque', device=DEFAULT_DEVICE),
    #     'niqe': pyiqa.create_metric('niqe', device=DEFAULT_DEVICE),
    #     'clipiqa': pyiqa.create_metric('clipiqa+', device=DEFAULT_DEVICE),
    #     'musiq-koniq': pyiqa.create_metric('musiq-koniq', device=DEFAULT_DEVICE),
    #     'topiq_nr': pyiqa.create_metric('topiq_nr', device=DEFAULT_DEVICE),
    # }
    #
    # # 2. Compute metrics
    # df = pd.DataFrame()
    # for metric_name, metric in metrics.items():
    #     metric_value = []
    #     for filepath in get_images_from_folder(folder):
    #         try:
    #             metric_value.append(metric(Image.open(filepath)).item())
    #         except Exception as e:
    #             print(f"File {filepath} was not processed correctly with metric {metric_name}, error {e}")
    #             metric_value.append(np.NaN)
    #     df[metric_name] = np.array(metric_value)
    #
    # # 3. Save csv
    # df['filename'] = [filepath.name.__str__() for filepath in get_images_from_folder(folder)]
    # df.to_csv(output_file, index=False)

# if __name__ == '__main__':
#
#     # Identify all folders
#     folders = get_generation_folders()
#
#     # Run on parallel on all folders
#     torch.multiprocessing.set_start_method('spawn')
#     process_map(run_on_folder, folders, max_workers=4)