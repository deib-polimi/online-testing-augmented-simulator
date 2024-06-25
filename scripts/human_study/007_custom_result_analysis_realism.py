import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.path_utils import PROJECT_DIR

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

    # Read all dfs:
    df = pd.concat([pd.read_csv(file, index_col=0)
     for file in
     PROJECT_DIR.joinpath('scripts', 'human_study', 'data', 'custom').glob("realism*.csv")])
    df['image'] = df['image'].map(lambda x: x.split("/")[1])
    df['class'] = df['image'].map(name_approach_map)
    df['class'] = df['class'].map(class_rename_dict)
    # df[['answer', 'class']]
    plt.figure(figsize=(15, 7.5))
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    font = {'size': 20}
    plt.rc('font', **font)
    sns.barplot(
        df, x='answer', y='class', capsize=.2, errorbar='ci', palette="light:white", edgecolor='black', lw=1,
        orient='h', hue='class',
        order=["Simulator", "Instruction-edited", "Inpainting", "Inpainting + \n Refining", "Real-World"])
    plt.xlabel("Realism Rating")
    plt.ylabel("Image Class")
    plt.xlim(1, 5)
    plt.tight_layout()
    # plt.show()
    plt.savefig("007_realism_custom.pdf")
