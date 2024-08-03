import itertools
import json
import re

import numpy as np
import pandas as pd

from domains.domain import DOMAIN_CATEGORIES_MAP
from domains.prompt import ALL_PROMPTS, PROMPT_TO_DOMAIN_MAP
from utils.path_utils import RESULT_DIR

if __name__ == '__main__':


    for approach, model in itertools.product([
        'stable_diffusion_inpainting',
    ],[
        'dave2', 'chauffeur', 'epoch', 'vit'
    ]
    ):

        error_count_domain = {}
        col_count_domain = {}
        oob_count_domain = {}
        unique_sectors_domain = {}
        max_cte_domain = {}
        mean_cte_domain = {}
        quality_domain = {}
        ood_domain = {}
        domain_category = []
        unique_sector = set()
        domain_list = []
        category_list = []
        sectors = set()
        runtime_domain = {}

        print(f"{approach}")

        for k, v in PROMPT_TO_DOMAIN_MAP.copy().items():
            PROMPT_TO_DOMAIN_MAP[re.sub('[^0-9a-zA-Z]+', '-', k)] = v

        domain_category = [DOMAIN_CATEGORIES_MAP[PROMPT_TO_DOMAIN_MAP[prompt]] for prompt in ALL_PROMPTS]
        domains = [PROMPT_TO_DOMAIN_MAP[prompt] for prompt in ALL_PROMPTS]

        for prompt in [
            "A-street-in-dust-storm-weather-photo-taken-from-a-car",
            "A-street-during-night-photo-taken-from-a-car",
            "A-street-in-forest-area-photo-taken-from-a-car",
            "A-street-in-summer-season-photo-taken-from-a-car",
            "A-street-during-afternoon-photo-taken-from-a-car",
            "A-street-in-sunny-weather-photo-taken-from-a-car",
            "A-street-in-winter-season-photo-taken-from-a-car",
            "A-street-in-autumn-season-photo-taken-from-a-car",
            "A-street-in-desert-area-photo-taken-from-a-car",
        ]:

            domain = PROMPT_TO_DOMAIN_MAP[prompt]
            run_name = f"online/cyclegan/{approach}/{model}/{re.sub('[^0-9a-zA-Z]+', '-', prompt)}"

            if not RESULT_DIR.joinpath(run_name, "info.json").exists():
                continue

            ood_value = pd.read_csv(RESULT_DIR.joinpath(run_name, "after", 'vae_reconstruction.csv'))['vae_0499'].mean()
            col_count_domain[domain] = sum([1 for x in json.load(open(RESULT_DIR.joinpath(run_name, "info.json")))['events'] if x.get('key', None) == 'collision'])
            oob_count_domain[domain] = sum([1 for x in json.load(open(RESULT_DIR.joinpath(run_name, "info.json")))['events'] if x.get('key', None) == 'out_of_track'])
            error_count_domain[domain] = oob_count_domain[domain] + col_count_domain[domain]

            df = pd.read_csv(RESULT_DIR.joinpath(run_name, "after", "log.csv"))
            orig_df = df.copy(deep=True)

            max_cte_domain[domain] = np.abs(df['cte']).max()
            mean_cte_domain[domain] = np.abs(df['cte']).mean()
            ood_domain[domain] = ood_value
            quality_domain[domain] = orig_df['steering_angle'].diff().abs().mean()
            runtime_domain[domain] = df['time'].max() - df['time'].min()

            unique_sector = set()
            ts = [int(x['timestamp']) for x in json.load(open(RESULT_DIR.joinpath(run_name, "info.json")))['events'] if 'timestamp' in x]

            df = df.set_index('time')

            for x in df.iloc[df.index.get_indexer(ts, method='nearest')]['sector'].unique():
                unique_sector.add(x)
            sectors = sectors.union(unique_sector)

            unique_sectors_domain[domain] = len(unique_sector)
            domain_list += [domain]
            category_list += [DOMAIN_CATEGORIES_MAP[PROMPT_TO_DOMAIN_MAP[prompt]]]

        err_df = pd.DataFrame.from_dict(data={
            'err': list(error_count_domain.values()),
            'col': list(col_count_domain.values()),
            'oob': list(oob_count_domain.values()),
            'ood': list(ood_domain.values()),
            'quality': list(quality_domain.values()),
            'max_cte': list(max_cte_domain.values()),
            'mean_cte': list(mean_cte_domain.values()),
            'unique_sectors': list(unique_sectors_domain.values()),
            'domain_category': category_list,
            'domain': domain_list,
            'runtime': list(runtime_domain.values()),
        })

        # print(err_df.sort_values('err'))
        # print(err_df.sort_values('err')[['col', 'oob', 'unique_sectors']].mean())
        # print(err_df.sort_values('err')[['col', 'oob', 'unique_sectors']].sum())
        # print(err_df[['mean_cte', 'quality']].mean())
        # print(err_df[['ood']].mean()) # Compute across all domains
        # print(len(sectors))
        # print(err_df[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors', 'quality']].max())
        # print(err_df[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors', 'quality']].mean())
        # print(err_df[['err', 'max_cte', 'mean_cte', 'col', 'oob', 'unique_sectors', 'quality']].sem())
        # print(err_df.groupby('domain_category')[['err']].agg(['min', 'mean', 'max', 'sem']))
        # print(err_df.groupby('domain_category')[['col']].agg(['min', 'mean', 'max', 'sem']))
        # print(err_df.groupby('domain_category')[['oob']].agg(['min', 'mean', 'max', 'sem']))
        # print(err_df.groupby('domain_category')[['quality']].agg(['min', 'mean', 'max', 'sem']))

        print((err_df.sort_values('err')['runtime'] / 60 / 1000).mean())
        print((err_df.sort_values('err')['runtime'] / 60 / 1000).sem())