import itertools
from math import sqrt

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils.path_utils import PROJECT_DIR
import numpy as np
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon


def cohend(d1, d2):
    """
    function to calculate Cohen's d for independent samples
    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    d = (u1 - u2) / s
    d = abs(d)

    result = ''
    if d < 0.2:
        result = 'negligible'
    if 0.2 <= d < 0.5:
        result = 'small'
    if 0.5 <= d < 0.8:
        result = 'medium'
    if d >= 0.8:
        result = 'large'

    return result, d


def run_wilcoxon_and_cohend(data1, data2):
    w_statistic, pvalue = wilcoxon(data1, data2, mode='exact')
    cohensd = cohend(data1, data2)
    # print(f"P-Value is: {pvalue}")
    # print(f"Cohen's D is: {cohensd}")

    return pvalue, cohensd


if __name__ == '__main__':

    df = pd.read_csv(PROJECT_DIR.joinpath('scripts', 'human_study', 'data', 'realism_log.csv'), index_col=0)
    name_approach_map = {}
    name_miou_map = {}
    for index, reason in enumerate(df['reason']):
        name_approach_map[f"{index:06d}_augmented.jpg"] = reason
    for index in range(19):
        name_approach_map[f"{index:06d}_real.jpg"] = "real-world"
        name_approach_map[f"{index:06d}_sim.jpg"] = "simulated"
        # name_approach_map[f"{index:06d}_sim.jpg"] = float(df[f'miou_{reason}'].values[index])

    class_rename_dict = {
        "simulated": "Simulator", 'instructpix2pix': "Instruction-edited", 'stable_diffusion_inpainting': "Inpainting",
        'stable_diffusion_inpainting_controlnet_refining': "Inpainting + \n Refining", 'real-world': "Real-World"
    }


    def read_file(path):
        df = pd.read_csv(path, index_col=0)
        df['username'] = str(path.name).split('_')[1]
        return df


    # Read all dfs:
    df = pd.concat([read_file(file)
                    for file in
                    PROJECT_DIR.joinpath('scripts', 'human_study', 'data', 'custom').glob("realism*.csv")])
    df['image'] = df['image'].map(lambda x: x.split("/")[1])
    df['class'] = df['image'].map(name_approach_map)
    df['class'] = df['class'].map(class_rename_dict)

    orig_df = df.copy(deep=True)

    data = pd.read_csv(
        "/home/banana/Downloads/Participant Expertise and Image Validation Survey.csv (1)/Participant Expertise and Image Validation Survey.csv",
        header=1,
        names=['time', 'username', 'has_license', 'license_years', 'ads_knowledge', 'ads_expertise', 'has_vision_issue',
               'vision_quality', 'ads_experience']
    )

    data = data.merge(df.groupby('username')['answer'].mean().reset_index(), on='username', how='left')

    df = data
    df['has_license'] = df['has_license'].map(lambda x: x == "Yes").astype(int)
    df['ads_knowledge'] = df['ads_knowledge'].map(lambda x: x == "Yes").astype(int)
    df['has_vision_issue'] = df['has_vision_issue'].map(lambda x: x == "Yes").astype(int)
    df['ads_experience'] = df['ads_experience'].map(lambda x: x == "Yes").astype(int)

    import statsmodels.api as sm
    # for col in set(data.columns).difference(['time', 'username']):
    X = df[['has_license', 'license_years', 'ads_knowledge', 'ads_expertise', 'has_vision_issue',
               'vision_quality', 'ads_experience']]
    y = df['answer']
    X = sm.add_constant(X)

    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    print(model.summary())