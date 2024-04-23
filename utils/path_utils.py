import pathlib
import itertools

import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).parent.parent.absolute()
RESULT_DIR = pathlib.Path("/media/banana/data/results/online-testing")
MODEL_DIR = pathlib.Path("/media/banana/data/models/online-testing")
LOG_DIR = pathlib.Path("/media/banana/data/logs/online-testing")


def get_images_from_folder(folder: pathlib.Path):
    return [x for x in sorted(list(folder.iterdir())) if x.suffix == '.jpg']


# Return all folders with generated images
def get_result_folders():
    result = []
    result += [
        folder.joinpath("before")
        for folder in itertools.chain(
            RESULT_DIR.joinpath("online", "stable_diffusion_inpainting").iterdir(),
            RESULT_DIR.joinpath("online", "stable_diffusion_inpainting_controlnet_refining").iterdir(),
            RESULT_DIR.joinpath("online", "instructpix2pix").iterdir(),
        )
    ]
    result += [
        folder.joinpath("after")
        for folder in itertools.chain(
            RESULT_DIR.joinpath("online", "stable_diffusion_inpainting").iterdir(),
            RESULT_DIR.joinpath("online", "stable_diffusion_inpainting_controlnet_refining").iterdir(),
            RESULT_DIR.joinpath("online", "instructpix2pix").iterdir(),
        )
    ]
    # Filter out all folders that are currently being generated
    result = sorted([r for r in result if r.joinpath("log.csv").exists()])
    return result

def get_result_folders_as_df():

    # 0. Initialization stuff
    df = pd.DataFrame()

    # 1A. Process InstructPix2Pix results
    for folder in RESULT_DIR.joinpath("online", "instructpix2pix").iterdir():
        dir_name = folder.name
        prompt = folder.name
        path = str(folder.absolute())
        path_before = str(folder.joinpath("before").absolute())
        path_after = str(folder.joinpath("after").absolute())
        n_images_before = 1
        n_images_after = 1

        if n_images_before != n_images_after:
            print(f"number of images in 'before' ({n_images_before}) and "
                  f"'after' ({n_images_after}) of {path} do not match.")

        if path_before.joinpath("log.csv"):
            print(f"logfile of {path_before} does not exists.")

        if path_after.joinpath("log.csv"):
            print(f"logfile of {path_after} does not exists.")