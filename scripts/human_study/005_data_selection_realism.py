import shutil

import pandas as pd
import torch
from tqdm import tqdm

from utils.path_utils import DATASET_DIR

if __name__ == '__main__':

    # Configuration settings
    image_directory = DATASET_DIR.joinpath("sampled_udacity_dataset")
    df = pd.read_csv(image_directory.joinpath("log.csv"), index_col=0)
    df = df[(df['domain'] == "lake_sunny_day")]
    target_directory = image_directory.joinpath("sampled_realism")
    target_directory.mkdir(parents=True, exist_ok=True)
    index_generator = range(len(df)).__iter__()
    sampled_df = pd.DataFrame()
    sample_size = 19

    with torch.no_grad():

        for approach in [
            "instructpix2pix",
            "stable_diffusion_inpainting",
            "stable_diffusion_inpainting_controlnet_refining",
            # "stable_diffusion_xl_inpainting",
            # "stable_diffusion_xl_inpainting_controlnet_refining",
        ]:
            more_df = df[df[f"miou_{approach}"] > 0.95]

            more_df = more_df.sample(n=min(sample_size, len(more_df)))

            more_df['reason'] = approach
            sampled_df = pd.concat([sampled_df, more_df])

        sampled_df = sampled_df.sample(frac=1)

        for img_filename, approach in tqdm(zip(sampled_df['image_filename'], sampled_df['reason'])):
            index = index_generator.__next__()
            source = image_directory.joinpath(approach, img_filename)
            destination = target_directory.joinpath(f"{index:06d}_augmented.jpg")
            shutil.copyfile(source, destination)

        sampled_df.to_csv(target_directory.joinpath("log.csv"))


        # Sample original images
        index_generator = range(len(df)).__iter__()
        sampled_df = sampled_df.sample(n=sample_size)

        for img_filename, approach in tqdm(zip(sampled_df['image_filename'], sampled_df['reason'])):
            index = index_generator.__next__()
            source = image_directory.joinpath("image", img_filename)
            destination = target_directory.joinpath(f"{index:06d}_sim.jpg")
            shutil.copyfile(source, destination)

        df = pd.DataFrame()
        image = []
        for x in range(57):
            image.append(f"{x:06d}_augmented.jpg")
        for x in range(18):
            image.append(f"{x:06d}_real.jpg")
        for x in range(18):
            image.append(f"{x:06d}_sim.jpg")
        df = df.sample(frac=1)
        df['image'] = image
        df.to_csv(target_directory.joinpath("turkfile.csv"))