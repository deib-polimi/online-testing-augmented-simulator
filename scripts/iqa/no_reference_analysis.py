import pathlib

import numpy as np
import pandas as pd
import pyiqa
import torch
from PIL import Image
from tqdm.contrib.concurrent import process_map
import seaborn
import matplotlib.pyplot as plt
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import get_images_from_folder, get_result_folders, get_result_folders_as_df


def run_on_one_folder(approach, domain, prompt, before_folder, after_folder):
    before_df = pd.read_csv(before_folder.joinpath("nr_iqa.csv")).fillna(0).sort_values(by="filename").set_index(
        "filename")
    after_df = pd.read_csv(after_folder.joinpath("nr_iqa.csv")).fillna(0).sort_values(by="filename").set_index(
        "filename")
    delta_df = (after_df - before_df)

    before_mean = before_df.mean(numeric_only=True)
    before_corr = before_df.corr(numeric_only=True)

    after_mean = after_df.mean(numeric_only=True)
    after_corr = after_df.corr(numeric_only=True)

    delta_mean = delta_df.mean(numeric_only=True)
    delta_corr = delta_df.corr(numeric_only=True)

    df = pd.DataFrame({
        'approach': approach,
        'domain': domain,
        'prompt': prompt,
        'before_mean': before_mean,
        'after_mean': after_mean,
        'delta_mean': delta_mean,
        # 'before_corr': before_corr,
        # 'after_corr': after_corr,
        # 'delta_corr': delta_corr,
    }).reset_index(names=["nr_iqa"])

    # seaborn.heatmap(before_corr)
    # plt.savefig("heatmap_before.pdf")
    # plt.clf()
    # plt.cla()
    # plt.close()
    #
    # seaborn.heatmap(after_corr)
    # plt.savefig("heatmap_after.pdf")
    # plt.clf()
    # plt.cla()
    # plt.close()

    return df


def worker_handler(args):
    return run_on_one_folder(**args)


if __name__ == '__main__':

    # Identify all folders
    df = get_result_folders_as_df()

    arg_list = [{
        'approach': approach,
        'domain': domain,
        'prompt': prompt,
        'before_folder': before_folder,
        'after_folder': after_folder,
    } for approach, domain, prompt, before_folder, after_folder
        in zip(df['approach'], df['domain'], df['dir_name'], df['path_before'], df['path_after'])]

    # worker_handler(arg_list[0])

    # Run on parallel on all folders
    # torch.multiprocessing.set_start_method('spawn')
    df = pd.concat(process_map(worker_handler, arg_list, max_workers=8))

    df.groupby(['nr_iqa', 'approach']).mean().to_csv("no_ref.csv")

    df['delta_percentage'] = df['delta_mean'] / df['before_mean'] * 100

    # TODO: work on the plotting part
    g = seaborn.boxplot(data=df, x='nr_iqa', y='delta_percentage', hue='approach')
    g.set(xlabel='Metric', ylabel='Percentage improvement (%)', ylim=(-100, 100))
    g.axhline(y=0, linewidth=2, color='red', ls=':')
    g.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')

    # plt.show()
    plt.tight_layout()
    plt.savefig("after.jpg")
    plt.clf()
    plt.cla()
    plt.close()

    # g = seaborn.boxplot(data=df, x='nr_iqa', y='delta_percentage')
    # g.set(xlabel='Metric', ylabel='Percentage improvement (%)', ylim=(-100, 100))
    # g.axhline(y=0, linewidth=2, color='red', ls=':')
    # g.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    #
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig("before.jpg")
