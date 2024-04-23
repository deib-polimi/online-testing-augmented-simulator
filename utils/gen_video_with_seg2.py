import pathlib
import numpy as np
import imageio
from tqdm import tqdm

from utils.path_utils import RESULT_DIR

IMAGEIO_DEFAULT_PLUGIN = "pyav"


def gen_video(folder: pathlib.Path):
    before_folder = folder.joinpath("before")
    after_folder = folder.joinpath("after")
    seg_before_folder = folder.joinpath("seg_before")
    seg_after_folder = folder.joinpath("seg_after")
    before_images = [x for x in sorted(list(before_folder.iterdir())) if x.suffix == '.jpg']
    after_images = [x for x in sorted(list(after_folder.iterdir())) if x.suffix == '.jpg']
    seg_before_images = [x for x in sorted(list(seg_before_folder.iterdir())) if x.suffix == '.jpg']
    seg_after_images = [x for x in sorted(list(seg_after_folder.iterdir())) if x.suffix == '.jpg']

    with imageio.v3.imopen(folder.joinpath("video_seg_2.mp4"), "w", plugin=IMAGEIO_DEFAULT_PLUGIN) as out_file:
        out_file.init_video_stream("hevc", fps=10, max_keyframe_interval=1000)
        for before, after, seg_before, seg_after in tqdm(
                zip(before_images, after_images, seg_before_images, seg_after_images)):
            before_image = imageio.v3.imread(before)
            after_image = imageio.v3.imread(after)
            seg_before_image = imageio.v3.imread(seg_before)
            seg_after_image = imageio.v3.imread(seg_after)
            image = np.concatenate(
                np.concatenate((before_image, after_image), axis=1),
                np.concatenate((seg_before_image, seg_after_image), axis=1),
                np.concatenate((before_image * 0.5 + seg_before_image * 0.5,
                                after_image * 0.5 + seg_after_image * 0.5), axis=1)
            )
            out_file.write_frame(image)


if __name__ == '__main__':
    segmentation_model = "clipseg"
    # folder = RESULT_DIR.joinpath("ip2p", "make_it_cloudy-2_5")
    folder = pathlib.Path("../log/gan_sunset_cow")
    # gen_video(folder)

    # Add plotting steering angle
    # TODO: remove duplicated code
    before_folder = folder.joinpath("before")
    after_folder = folder.joinpath("after")
    seg_before_folder = folder.joinpath("seg_before_2", segmentation_model)
    seg_after_folder = folder.joinpath("seg_after_2", segmentation_model)
    before_images = [x for x in sorted(list(before_folder.iterdir())) if x.suffix == '.jpg']
    after_images = [x for x in sorted(list(after_folder.iterdir())) if x.suffix == '.jpg']
    seg_before_images = [x for x in sorted(list(seg_before_folder.iterdir())) if x.suffix == '.jpg']
    seg_after_images = [x for x in sorted(list(seg_after_folder.iterdir())) if x.suffix == '.jpg']

    with imageio.v3.imopen(folder.joinpath(f"video_{segmentation_model}2.mp4"), "w",
                           plugin=IMAGEIO_DEFAULT_PLUGIN) as out_file:
        out_file.init_video_stream("hevc", fps=10, max_keyframe_interval=1000)
        for before, after, seg_before, seg_after in tqdm(
                zip(before_images, after_images, seg_before_images, seg_after_images)):
            before_image = imageio.v3.imread(before)
            after_image = imageio.v3.imread(after)
            seg_before_image = imageio.v3.imread(seg_before)
            seg_after_image = imageio.v3.imread(seg_after)
            image = np.concatenate([
                np.concatenate((before_image, after_image), axis=1),
                np.concatenate((seg_before_image, seg_after_image), axis=1),
                np.concatenate((before_image * 0.5 + seg_before_image * 0.5,
                                after_image * 0.5 + seg_after_image * 0.5), axis=1),
            ])
            out_file.write_frame(image)
