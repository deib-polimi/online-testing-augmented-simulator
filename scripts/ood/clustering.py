import pandas as pd

from domains.domain import DOMAIN_CATEGORIES
from utils.path_utils import get_result_folders_as_df

if __name__ == '__main__':
    # Get all runs
    df = get_result_folders_as_df()

    # Filter out invalid domains
    # df = df[df['domain'].map(lambda x: x in DOMAIN_CATEGORIES)]

    # for model in ['dave2', 'chauffeur', 'epoch']:
    df['rec'] = [
        pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(by="filename").set_index(
            "filename").mean()[0] if after_folder.joinpath("vae_reconstruction.csv").exists() else pd.NA
        for after_folder in df['path_after']]

    df = df.dropna()

    in_dist = {}
    mid_dist = {}
    out_dist = {}

    for approach in [
        'instructpix2pix',
        'stable_diffusion_inpainting',
        'stable_diffusion_inpainting_controlnet_refining',
    ]:
        df = df.groupby(['domain', 'approach'])['rec'].mean().reset_index()
        sub_df = df[df['approach'] == approach]
        in_dist[approach] = set(sub_df.sort_values('rec')['domain'][:int(len(sub_df)*1/3)])
        mid_dist[approach] = set(sub_df.sort_values('rec')['domain'][int(len(sub_df)*1/3):int(len(sub_df)*2/3)])
        out_dist[approach] = set(sub_df.sort_values('rec')['domain'][int(len(sub_df)*2/3):int(len(sub_df)*3/3)])

    for dist in [
        in_dist, mid_dist, out_dist
    ]:
        min_set = set(df['domain'])
        for k, v in dist.items():
            min_set = min_set.intersection(v)
            print(v)
            print(min_set)

    # df_dave = df[df['model'] == 'dave2']
    # df_chauffeur = df[df['model'] == 'chauffeur']
    # df_epoch = df[df['model'] == 'epoch']
