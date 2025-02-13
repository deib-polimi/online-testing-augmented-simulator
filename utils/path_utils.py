import pathlib
import itertools
import os
import pandas as pd

from domains.domain import DOMAIN_CATEGORIES_MAP
from domains.instruction import ALL_INSTRUCTIONS_FOLDER_MAP

from domains.prompt import ALL_PROMPTS_FOLDER_MAP

BASE_DIR = os.environ.get("BASE_DIR", "media/banana/data")
PROJECT_DIR = pathlib.Path(__file__).parent.parent.absolute()
RESULT_DIR = pathlib.Path(BASE_DIR, "results/online-testing")
MODEL_DIR = pathlib.Path(BASE_DIR, "models/online-testing")
LOG_DIR = pathlib.Path(BASE_DIR, "logs/online-testing")
DATASET_DIR = pathlib.Path(BASE_DIR, "dataset")

for directory in [RESULT_DIR, MODEL_DIR, LOG_DIR, DATASET_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def get_images_from_folder(folder: pathlib.Path):
    return [x for x in sorted(list(folder.iterdir())) if x.suffix == '.jpg']


# Return all folders with generated images
def get_result_folders():
    result = []
    folders =  sum(
        [list(RESULT_DIR.joinpath("online", "cyclegan", approach, model).iterdir())
        for approach, model in itertools.product(
            [
             "instructpix2pix",
             "stable_diffusion_inpainting",
             "stable_diffusion_inpainting_controlnet_refining",
             ],
            ["dave2", "chauffeur", "epoch", "vit"],
        )], [])
    result += [r.joinpath('before') for r in folders]
    result += [r.joinpath('after') for r in folders]

    # Filter out all folders that are currently being generated
    result = sorted([r for r in result if r.joinpath("log.csv").exists()])
    return result


def get_result_folders_as_df():
    # 0. Initialization stuff
    dir_name_list = []
    domain_list = []
    domain_category_list = []
    path_list = []
    path_before_list = []
    path_after_list = []
    approach_list = []
    model_name_list = []

    # 1A. Process InstructPix2Pix results
    mapping = {v: k for k, v in ALL_INSTRUCTIONS_FOLDER_MAP.items()}
    for model_name in ['dave2', 'chauffeur', 'epoch']:
        for folder in sorted(list(RESULT_DIR.joinpath("online", "cyclegan", "instructpix2pix", model_name).iterdir())):
            dir_name = folder.name
            domain = mapping[dir_name]
            domain_category = DOMAIN_CATEGORIES_MAP[domain]
            path = str(folder.absolute())
            path_before = folder.joinpath("before").absolute()
            path_after = folder.joinpath("after").absolute()
            n_images_before = len([x for x in sorted(list(path_before.iterdir())) if x.suffix == '.jpg'])
            n_images_after = len([x for x in sorted(list(path_after.iterdir())) if x.suffix == '.jpg'])
            approach = "instructpix2pix"

            if n_images_before != n_images_after:
                print(f"number of images in 'before' ({n_images_before}) and "
                      f"'after' ({n_images_after}) of {path} do not match.")
                continue

            if not path_before.joinpath("log.csv").exists():
                print(f"logfile of {path_before} does not exists.")
                continue

            if not path_after.joinpath("log.csv").exists():
                print(f"logfile of {path_after} does not exists.")
                continue

            dir_name_list += [dir_name]
            domain_list += [domain]
            domain_category_list += [domain_category]
            path_list += [path]
            path_before_list += [path_before]
            path_after_list += [path_after]
            approach_list += [approach]
            model_name_list += [model_name]


    # 1B. Process Stable Diffusion Inpainting results
    mapping = {v: k for k, v in ALL_PROMPTS_FOLDER_MAP.items()}
    for model_name in ['dave2', 'chauffeur', 'epoch']:
        for folder in sorted(list(RESULT_DIR.joinpath("online", "cyclegan", "stable_diffusion_inpainting", model_name).iterdir())):
            dir_name = folder.name
            if dir_name not in mapping.keys():
                continue
            domain = mapping[dir_name]
            domain_category = DOMAIN_CATEGORIES_MAP[domain]
            path = str(folder.absolute())
            path_before = folder.joinpath("before").absolute()
            path_after = folder.joinpath("after").absolute()
            n_images_before = len([x for x in sorted(list(path_before.iterdir())) if x.suffix == '.jpg'])
            n_images_after = len([x for x in sorted(list(path_after.iterdir())) if x.suffix == '.jpg'])
            approach = "stable_diffusion_inpainting"

            if n_images_before != n_images_after:
                print(f"number of images in 'before' ({n_images_before}) and "
                      f"'after' ({n_images_after}) of {path} do not match.")
                continue

            if not path_before.joinpath("log.csv").exists():
                print(f"logfile of {path_before} does not exists.")
                continue

            if not path_after.joinpath("log.csv").exists():
                print(f"logfile of {path_after} does not exists.")
                continue

            dir_name_list += [dir_name]
            domain_list += [domain]
            domain_category_list += [domain_category]
            path_list += [path]
            path_before_list += [path_before]
            path_after_list += [path_after]
            approach_list += [approach]
            model_name_list += [model_name]

    # 1C. Process Stable Diffusion Inpainting Controlnet Refining results
    mapping = {v: k for k, v in ALL_PROMPTS_FOLDER_MAP.items()}
    for model_name in ['dave2', 'chauffeur', 'epoch']:
        for folder in sorted(
                list(RESULT_DIR.joinpath("online", "cyclegan", "stable_diffusion_inpainting_controlnet_refining", model_name).iterdir())):
            dir_name = folder.name
            if dir_name not in mapping.keys():
                continue
            domain = mapping[dir_name]
            domain_category = DOMAIN_CATEGORIES_MAP[domain]
            path = str(folder.absolute())
            path_before = folder.joinpath("before").absolute()
            path_after = folder.joinpath("after").absolute()
            n_images_before = len([x for x in sorted(list(path_before.iterdir())) if x.suffix == '.jpg'])
            n_images_after = len([x for x in sorted(list(path_after.iterdir())) if x.suffix == '.jpg'])
            approach = "stable_diffusion_inpainting_controlnet_refining"

            if n_images_before != n_images_after:
                print(f"number of images in 'before' ({n_images_before}) and "
                      f"'after' ({n_images_after}) of {path} do not match.")
                continue

            if not path_before.joinpath("log.csv").exists():
                print(f"logfile of {path_before} does not exists.")
                continue

            if not path_after.joinpath("log.csv").exists():
                print(f"logfile of {path_after} does not exists.")

            dir_name_list += [dir_name]
            domain_list += [domain]
            domain_category_list += [domain_category]
            path_list += [path]
            path_before_list += [path_before]
            path_after_list += [path_after]
            approach_list += [approach]
            model_name_list += [model_name]

    return pd.DataFrame({
        'dir_name': dir_name_list,
        'domain': domain_list,
        'domain_category': domain_category_list,
        'path': path_list,
        'path_before': path_before_list,
        'path_after': path_after_list,
        'approach': approach_list,
        'model': model_name_list,
    })
