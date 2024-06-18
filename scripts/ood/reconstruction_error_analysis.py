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
from utils.path_utils import get_images_from_folder, get_result_folders, get_result_folders_as_df, RESULT_DIR
from scipy.stats import gamma

plt.rcParams["figure.figsize"] = (12, 8)

def run_on_one_folder(approach, domain, prompt, before_folder, after_folder):
    before_df = pd.read_csv(before_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
        by="filename").set_index(
        "filename")
    after_df = pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
        by="filename").set_index(
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

    return df


def worker_handler(args):
    return run_on_one_folder(**args)


def analyze_domain(args):
    output_folder = RESULT_DIR.joinpath("reconstruction_error_analysis", "domain")

    def handle(approach, domain, prompt, before_folder, after_folder):
        output_file = output_folder.joinpath(approach, f"{domain}.pdf")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        before_df = pd.read_csv(before_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
            by="filename").set_index(
            "filename")
        after_df = pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
            by="filename").set_index(
            "filename")

        shape, loc, scale = gamma.fit(before_df['vae_0499'], floc=0)
        x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                        gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
        plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label='nominal')

        shape, loc, scale = gamma.fit(after_df['vae_0499'], floc=0)
        x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                        gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
        plt.ylabel("samples")
        plt.xlabel("reconstruction error")
        plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f"{domain} ({approach})")

        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.clf()
        plt.cla()
        plt.close()

    handle(**args)

def compare_tail_domain(args):
    output_folder = RESULT_DIR.joinpath("reconstruction_error_analysis", "domain")

    def handle(approach, domain, prompt, before_folder, after_folder):
        output_file = output_folder.joinpath(approach, f"{domain}.pdf")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        before_df = pd.read_csv(before_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
            by="filename").set_index("filename")
        after_df = pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
            by="filename").set_index("filename")

        df = pd.DataFrame()
        df['before'] = before_df['vae_0499']
        df['after'] = after_df['vae_0499']
        # corr = df.corr(method="spearman", numeric_only=True)
        # return pd.DataFrame({
        #     'approach': approach,
        #     'domain': domain,
        #     'prompt': prompt,
        #     'corr': [corr['before'].values[1]],
        # })
        before_indexes = set(df.sort_values('before')[int(len(df)*0.95):].index)
        after_indexes = set(df.sort_values('after')[int(len(df)*0.95):].index)
        return pd.DataFrame({
            'approach': approach,
            'domain': domain,
            'prompt': prompt,
            'intersection': [len(after_indexes.intersection(before_indexes))/len(before_indexes)*100],
        })
        # return corr

    return handle(**args)

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

    # approaches = df['approach'].unique().tolist()
    # domains = df['domain'].unique().tolist()
    # domain_categories = df['domain_category'].unique().tolist()
    # # domain_categories = df['approach'].unique().tolist()
    #
    # # 0001. Plot reconstruction errors for every run
    # process_map(analyze_domain, arg_list, max_workers=8)
    #
    # # 0002. Analyze by approach
    # output_folder = RESULT_DIR.joinpath("reconstruction_error_analysis", "approach")
    # for approach in approaches:
    #     output_file = output_folder.joinpath(f"{approach}.pdf")
    #     output_file.parent.mkdir(parents=True, exist_ok=True)
    #     sub_df = df[df['approach'] == approach]
    #     iterator = zip(sub_df['approach'], sub_df['domain'], sub_df['dir_name'],
    #                    sub_df['path_before'], sub_df['path_after'])
    #     first_flag = True
    #     for approach, domain, prompt, before_folder, after_folder in iterator:
    #         if first_flag:
    #             before_df = pd.read_csv(before_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
    #                 by="filename").set_index("filename")
    #
    #             shape, loc, scale = gamma.fit(before_df['vae_0499'], floc=0)
    #             x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
    #                             gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
    #             plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label='nominal')
    #             first_flag = False
    #
    #         after_df = pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
    #             by="filename").set_index(
    #             "filename")
    #         shape, loc, scale = gamma.fit(after_df['vae_0499'], floc=0)
    #         x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
    #                         gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
    #         plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f"{domain}")
    #
    #     plt.ylabel("samples")
    #     plt.xlabel("reconstruction error")
    #     plt.xlim([0, 0.4])
    #     plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncols=5)
    #     plt.tight_layout()
    #     plt.savefig(output_file)
    #     plt.clf()
    #     plt.cla()
    #     plt.close()
    #
    # # 0003. Analyze by domain
    # output_folder = RESULT_DIR.joinpath("reconstruction_error_analysis", "domain")
    # for domain in domains:
    #     output_file = output_folder.joinpath(f"{domain}.pdf")
    #     output_file.parent.mkdir(parents=True, exist_ok=True)
    #     sub_df = df[df['domain'] == domain]
    #     iterator = zip(sub_df['approach'], sub_df['domain'], sub_df['dir_name'],
    #                    sub_df['path_before'], sub_df['path_after'])
    #     first_flag = True
    #     for approach, domain, prompt, before_folder, after_folder in iterator:
    #         if first_flag:
    #             before_df = pd.read_csv(before_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
    #                 by="filename").set_index("filename")
    #
    #             shape, loc, scale = gamma.fit(before_df['vae_0499'], floc=0)
    #             x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
    #                             gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
    #             plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label='nominal')
    #             first_flag = False
    #
    #         after_df = pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
    #             by="filename").set_index(
    #             "filename")
    #         shape, loc, scale = gamma.fit(after_df['vae_0499'], floc=0)
    #         x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
    #                         gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
    #         plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f"{approach}")
    #
    #     plt.ylabel("samples")
    #     plt.xlabel("reconstruction error")
    #     plt.xlim([0, 0.4])
    #     plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncols=4)
    #     plt.tight_layout()
    #     plt.savefig(output_file)
    #     plt.clf()
    #     plt.cla()
    #     plt.close()
    #
    # # 0004. Analyze by domain category and approach
    # output_folder = RESULT_DIR.joinpath("reconstruction_error_analysis", "approach")
    # for domain_category in domain_categories:
    #     for approach in approaches:
    #         output_file = output_folder.joinpath(domain_category, f"{approach}.pdf")
    #         output_file.parent.mkdir(parents=True, exist_ok=True)
    #         sub_df = df[df['domain_category'] == domain_category]
    #         sub_df = sub_df[sub_df['approach'] == approach]
    #         iterator = zip(sub_df['approach'], sub_df['domain'], sub_df['dir_name'],
    #                        sub_df['path_before'], sub_df['path_after'])
    #         first_flag = True
    #         for approach, domain, prompt, before_folder, after_folder in iterator:
    #             if first_flag:
    #                 before_df = pd.read_csv(before_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
    #                     by="filename").set_index("filename")
    #
    #                 shape, loc, scale = gamma.fit(before_df['vae_0499'], floc=0)
    #                 x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
    #                                 gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
    #                 plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label='nominal')
    #                 first_flag = False
    #
    #             after_df = pd.read_csv(after_folder.joinpath("vae_reconstruction.csv")).fillna(0).sort_values(
    #                 by="filename").set_index(
    #                 "filename")
    #             shape, loc, scale = gamma.fit(after_df['vae_0499'], floc=0)
    #             x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
    #                             gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
    #             plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f"{domain}")
    #
    #         plt.ylabel("samples")
    #         plt.xlabel("reconstruction error")
    #         plt.xlim([0, 0.4])
    #         plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncols=4)
    #         plt.tight_layout()
    #         plt.savefig(output_file)
    #         plt.clf()
    #         plt.cla()
    #         plt.close()

    # 0005. Compare tail distribution
    df = pd.concat(process_map(compare_tail_domain, arg_list, max_workers=8))

        # Run on parallel on all folders
        # torch.multiprocessing.set_start_method('spawn')
        # df = pd.concat(process_map(worker_handler, arg_list, max_workers=8))

        # df.groupby(['nr_iqa', 'approach']).mean().to_csv("no_ref.csv")
        #
        # df['delta_percentage'] = df['delta_mean'] / df['before_mean'] * 100
        #
        # # TODO: work on the plotting part
        # g = seaborn.boxplot(data=df, x='nr_iqa', y='delta_percentage', hue='approach')
        # g.set(xlabel='Metric', ylabel='Percentage improvement (%)')
        # g.axhline(y=0, linewidth=2, color='red', ls=':')
        # plt.savefig("temp.jpg")
