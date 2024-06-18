import random
import pandas as pd
import shutil

from tqdm import tqdm

from utils.path_utils import DATASET_DIR

if __name__ == '__main__':

    # Configuration settings
    n_samples = 6000
    target_directory = DATASET_DIR.joinpath("sampled_udacity_dataset")

    # Initialization settings
    random.seed(42)
    df = pd.DataFrame()
    target_directory.mkdir(parents=True, exist_ok=True)
    target_directory.joinpath("image").mkdir(parents=True, exist_ok=True)
    target_directory.joinpath("segmentation").mkdir(parents=True, exist_ok=True)

    # Generate dataset and get domains
    dataset_dir = DATASET_DIR.joinpath('udacity_dataset')
    domain_dir_list = [
        DATASET_DIR.joinpath('udacity_dataset', 'lake_sunny_day'),
        # DATASET_DIR.joinpath('udacity_dataset', 'lake_sunny_daynight'),
        DATASET_DIR.joinpath('udacity_dataset', 'mountain_sunny_day'),
        # DATASET_DIR.joinpath('udacity_dataset', 'mountain_sunny_daynight'),
    ]

    # Get `n_samples` from the each domain
    for domain_dir in domain_dir_list:
        domain_df = pd.read_csv(domain_dir.joinpath("log.csv"))
        domain_df = domain_df.sample(n=n_samples)
        domain_df['domain'] = domain_dir.name
        df = pd.concat([df, domain_df])

    # Generate new filenames
    for img_filename, domain in tqdm(zip(df['image_filename'], df['domain'])):
        source = dataset_dir.joinpath(domain, "image", img_filename)
        destination = target_directory.joinpath("image", img_filename)
        shutil.copyfile(source, destination)
    for seg_filename, domain in tqdm(zip(df['segmentation_filename'], df['domain'])):
        source = dataset_dir.joinpath(domain, "segmentation", seg_filename)
        destination = target_directory.joinpath("segmentation", seg_filename)
        shutil.copyfile(source, destination)

    # Save dataframe to csv
    df.to_csv(target_directory.joinpath("log.csv"))