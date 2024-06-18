import random
import shutil

import pandas as pd
import torch
from tqdm import tqdm

from utils.path_utils import DATASET_DIR

if __name__ == '__main__':

    # Configuration settings
    image_directory = DATASET_DIR.joinpath("sampled_udacity_dataset")
    df = pd.read_csv(image_directory.joinpath("log.csv"), index_col=0)
    df = df[(df['domain'] == "mountain_sunny_day") | (df['domain'] == "lake_sunny_day")]
    # df = df[(df['domain'] == "lake_sunny_day")]
    target_directory = image_directory.joinpath("sampled_semantic")
    target_directory.mkdir(parents=True, exist_ok=True)
    index_generator = range(len(df)).__iter__()
    sampled_df = pd.DataFrame()

    with torch.no_grad():

        for approach in [
            "instructpix2pix",
            "stable_diffusion_inpainting",
            "stable_diffusion_inpainting_controlnet_refining",
            # "stable_diffusion_xl_inpainting",
            # "stable_diffusion_xl_inpainting_controlnet_refining",
        ]:
            less_df = df[df[f"miou_{approach}"] <= 0.905]
            more_df = df[df[f"miou_{approach}"] > 0.905]

            print(f"{approach} has {len(less_df)} invalid generation")
            print(f"{approach} has {len(more_df)} valid generation")

            s_less_df = less_df[(less_df['domain'] == "lake_sunny_day")].sample(n=min(14, len(less_df[(less_df['domain'] == "lake_sunny_day")])))
            s_more_df = more_df[(more_df['domain'] == "lake_sunny_day")].sample(n=min(14, len(more_df[(more_df['domain'] == "lake_sunny_day")])))

            m_less_df = less_df[(less_df['domain'] == "mountain_sunny_day")].sample(n=min(18-len(s_less_df), len(less_df[(less_df['domain'] == "mountain_sunny_day")])))
            m_more_df = more_df[(more_df['domain'] == "mountain_sunny_day")].sample(n=min(18-len(s_more_df), len(more_df[(more_df['domain'] == "mountain_sunny_day")])))

            s_less_df['reason'] = approach
            s_more_df['reason'] = approach
            m_less_df['reason'] = approach
            m_more_df['reason'] = approach

            sampled_df = pd.concat([sampled_df, s_less_df, s_more_df, m_less_df, m_more_df])

        sampled_df = sampled_df.sample(frac=1)

        for img_filename, approach in tqdm(zip(sampled_df['image_filename'], sampled_df['reason'])):
            index = index_generator.__next__()
            source = image_directory.joinpath(approach, img_filename)
            destination = target_directory.joinpath(f"{index:06d}_augmented.jpg")
            shutil.copyfile(source, destination)
            source = image_directory.joinpath("image", img_filename)
            destination = target_directory.joinpath(f"{index:06d}_original.jpg")
            shutil.copyfile(source, destination)
        sampled_df.to_csv(target_directory.joinpath("log.csv"))

        df = pd.DataFrame()
        image_a = []
        image_b = []
        for x in range(108):
            if random.random() > 0.5:
                image_a.append(f"{x:06d}_augmented.jpg")
                image_b.append(f"{x:06d}_original.jpg")
            else:
                image_b.append(f"{x:06d}_augmented.jpg")
                image_a.append(f"{x:06d}_original.jpg")
        image_a.append("yes_augmented.jpg")
        image_a.append("no_augmented.jpg")
        image_b.append("yes_original.jpg")
        image_b.append("no_original.jpg")
        df['image_a'] = image_a
        df['image_b'] = image_b
        df = df.sample(frac=1)
        df.to_csv(target_directory.joinpath("turkfile.csv"))