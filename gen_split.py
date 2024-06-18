import re

import numpy as np
import shutil

import pandas as pd

from domains.domain import DOMAIN_CATEGORIES_MAP
from domains.prompt import WEATHER_PROMPTS, SEASON_PROMPTS, COUNTRY_PROMPTS, CITY_PROMPTS, LOCATION_PROMPTS, \
    TIME_PROMPTS
from utils.path_utils import RESULT_DIR

if __name__ == '__main__':

    curr_counter = 18
    df = pd.DataFrame()

    for approach in ["stable_diffusion_inpainting", "stable_diffusion_inpainting_controlnet_refining"]:

        for prompt_set in [WEATHER_PROMPTS, SEASON_PROMPTS, COUNTRY_PROMPTS, CITY_PROMPTS, LOCATION_PROMPTS, TIME_PROMPTS]:

            for _ in range(3):

                prompt = np.random.choice(prompt_set, 1)[0]

                directory = RESULT_DIR.joinpath("online", approach, re.sub('[^0-9a-zA-Z]+', '-', prompt))
                if not directory.exists():
                    continue
                filename_after = np.random.choice(list(directory.joinpath("after").glob("*.jpg")), 1)[0]
                filename_before = directory.joinpath("before", filename_after.name)

                shutil.copy(filename_before, f"{curr_counter:03d}_b.jpg")
                shutil.copy(filename_after, f"{curr_counter:03d}_a.jpg")
                curr_counter += 1
                df = pd.concat([df,
                                pd.DataFrame({
                                    'filename': [filename_after.name],
                                    'approach': [approach],
                                    'prompt': [prompt],
                                })])
    df.to_csv("gen_split.csv")



