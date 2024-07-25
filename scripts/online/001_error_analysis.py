import json
import re

import numpy as np
import pandas as pd

from domains.domain import DOMAIN_CATEGORIES_MAP
from domains.prompt import ALL_PROMPTS, PROMPT_TO_DOMAIN_MAP
from utils.path_utils import RESULT_DIR

if __name__ == '__main__':

    error_count_domain = {}
    col_count_domain = {}
    oob_count_domain = {}
    unique_sectors_domain = {}
    max_cte_domain = {}
    mean_cte_domain = {}
    quality_domain = {}
    domain_category = []
    unique_sector = set()

    for approach in [
        'stable_diffusion_inpainting',
        'stable_diffusion_inpainting_controlnet_refining'
    ]:

        print(f"{approach}")

        domain_category = [DOMAIN_CATEGORIES_MAP[PROMPT_TO_DOMAIN_MAP[prompt]] for prompt in ALL_PROMPTS]
        domains = [PROMPT_TO_DOMAIN_MAP[prompt] for prompt in ALL_PROMPTS]

        for prompt in ALL_PROMPTS:

            domain = PROMPT_TO_DOMAIN_MAP[prompt]
            run_name = f"online/{approach}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"
            ood_value = pd.read_csv(RESULT_DIR.joinpath(run_name, "after", 'vae_reconstruction.csv'))
            errors = sum(json.load(open(RESULT_DIR.joinpath(run_name, "info.json")))['episode_metrics'].values())
            error_count_domain[domain] = errors
            col_count_domain[domain] = json.load(open(RESULT_DIR.joinpath(run_name, "info.json")))['episode_metrics'].get('collisionCount', 0)
            oob_count_domain[domain] = json.load(open(RESULT_DIR.joinpath(run_name, "info.json")))['episode_metrics'].get('outOfTrackCount', 0)

            df = pd.read_csv(RESULT_DIR.joinpath(run_name, "after", "log.csv"))
            orig_df = df.copy(deep=True)

            max_cte_domain[domain] = np.abs(df['cte']).max()
            mean_cte_domain[domain] = np.abs(df['cte']).mean()

            quality_domain[domain] = orig_df['steering_angle'].diff().abs().mean()

            df['sector_gap'] = np.append([0], ((df['sector'][1:].to_numpy() -df['sector'][:-1].to_numpy()) % 40))
            df = df[df['sector_gap'] == 2]

            # print(len(df), prompt)

            for x in df['sector'].unique():
                unique_sector.add(x)

            # print(df['sector'].unique())
            unique_sectors_domain[domain] = len(df['sector'].unique())
            # break

        # print(unique_sector)
        # print(len(unique_sector))

        err_df = pd.DataFrame.from_dict(data={
            'err': list(error_count_domain.values()),
            'col': list(col_count_domain.values()),
            'oob': list(oob_count_domain.values()),
            'quality': list(quality_domain.values()),
            'max_cte': list(max_cte_domain.values()),
            'mean_cte': list(mean_cte_domain.values()),
            'unique_sectors': list(unique_sectors_domain.values()),
            # 'unique_sectors': len(unique_sector),
            # 'sectors': unique_sector,
            'domain_category': domain_category,
            'domain': domains
        })
        # max_cte_df = pd.DataFrame.from_dict(data={'value': list(max_cte_domain.values()), 'domain_category': domain_category, 'domain': domains})
        # mean_cte_df = pd.DataFrame.from_dict(data={'value': list(mean_cte_domain.values()), 'domain_category': domain_category, 'domain': domains})

        print(err_df.sort_values('err'))
        print(err_df[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors', 'quality']].max())
        print(err_df[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors', 'quality']].mean())
        print(err_df[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors', 'quality']].sem())
        # print(err_df.groupby('domain_category')[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors']].agg(['min', 'mean', 'max', 'sem']))
        print(err_df.groupby('domain_category')[['err']].agg(['min', 'mean', 'max', 'sem']))
        print(err_df.groupby('domain_category')[['col']].agg(['min', 'mean', 'max', 'sem']))
        print(err_df.groupby('domain_category')[['oob']].agg(['min', 'mean', 'max', 'sem']))
        print(err_df.groupby('domain_category')[['quality']].agg(['min', 'mean', 'max', 'sem']))

        # print(max_cte_df.sort_values(0))
        # print(mean_cte_df.sort_values(0))
        #
        # print(err_df.mean())
        # print(max_cte_df.mean())
        # print(mean_cte_df.mean())