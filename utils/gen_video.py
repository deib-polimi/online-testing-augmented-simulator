import pathlib
import numpy as np
import imageio
from tqdm import tqdm

from utils.path_utils import RESULT_DIR

IMAGEIO_DEFAULT_PLUGIN = "pyav"


def gen_video(folder: pathlib.Path):
    before_folder = folder.joinpath("before")
    after_folder = folder.joinpath("after")
    before_images = [x for x in sorted(list(before_folder.iterdir())) if x.suffix == '.jpg']
    after_images = [x for x in sorted(list(after_folder.iterdir())) if x.suffix == '.jpg']



    with imageio.v3.imopen(folder.joinpath("video.mp4"), "w", plugin=IMAGEIO_DEFAULT_PLUGIN) as out_file:
        out_file.init_video_stream("hevc", fps=10, max_keyframe_interval=1000)
        for before, after in tqdm(zip(before_images, after_images)):
            before_image = imageio.v3.imread(before)
            after_image = imageio.v3.imread(after)
            image = np.concatenate((before_image, after_image), axis=1)
            out_file.write_frame(image)


if __name__ == '__main__':
    folder = pathlib.Path("/media/banana/data/results/online-testing/online/cyclegan/stable_diffusion_inpainting_controlnet_refining/chauffeur/A-street-during-night-photo-taken-from-a-car")
    # folder = RESULT_DIR.joinpath("sd_inpainting_sd", "a_street_in_the_usa-10")
    # gen_video(folder)

    # Add plotting steering angle
    before_folder = folder.joinpath("before/image")
    after_folder = folder.joinpath("after/image")
    before_images = [x for x in sorted(list(before_folder.iterdir())) if x.suffix == '.jpg']
    after_images = [x for x in sorted(list(after_folder.iterdir())) if x.suffix == '.jpg']

    if folder.joinpath("video.mp4").exists():
        exit(0)

    with imageio.v3.imopen(folder.joinpath("video.mp4"), "w", plugin=IMAGEIO_DEFAULT_PLUGIN) as out_file:
        out_file.init_video_stream("hevc", fps=30, max_keyframe_interval=1000)
        for before, after in tqdm(zip(before_images, after_images)):
            before_image = imageio.v3.imread(before)
            after_image = imageio.v3.imread(after)
            image = np.concatenate((before_image, after_image), axis=1)
            out_file.write_frame(image)
