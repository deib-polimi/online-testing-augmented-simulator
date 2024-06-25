import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_score
from utils.path_utils import PROJECT_DIR

if __name__ == '__main__':

    df = pd.read_csv(PROJECT_DIR.joinpath('scripts', 'human_study', 'data', 'semantic_log.csv'), index_col=0)
    name_approach_map = {}
    name_miou_map = {}
    for index, reason in enumerate(df['reason']):
        name_approach_map[f"{index:06d}_augmented.jpg"] = reason
        name_approach_map[f"{index:06d}_original.jpg"] = reason
        name_miou_map[f"{index:06d}_augmented.jpg"] = float(df[f'miou_{reason}'].values[index])
        name_miou_map[f"{index:06d}_original.jpg"] = float(df[f'miou_{reason}'].values[index])
    for index in range(19):
        name_approach_map[f"{index:06d}_real.jpg"] = "real-world"
        name_approach_map[f"{index:06d}_sim.jpg"] = "simulated"

    class_rename_dict = {
        "simulated": "Simulator", 'instructpix2pix': "Instruction-edited", 'stable_diffusion_inpainting': "Inpainting",
        'stable_diffusion_inpainting_controlnet_refining': "Inpainting + \n Refining", 'real-world': "Real-World"
    }

    df = pd.read_csv(PROJECT_DIR.joinpath('scripts', 'human_study', 'data', 'semantic_mturk_data.csv'), index_col=0)
    df['image'] = df['Input.image_a']
    df['class'] = df['image'].map(name_approach_map)
    df['class'] = df['class'].map(class_rename_dict)
    df['miou'] = df['image'].map(name_miou_map)
    df['true'] = df['miou'] >= 0.9
    df['answer'] = df['Answer.answer']
    df['pred'] = df['Answer.answer'] == 'yes'
    df = df[df['AssignmentStatus'] != "Rejected"]
    df = df[~df['class'].isna()]

    orig_df = df.copy(deep=True)

    for df in [
        orig_df,
        orig_df[orig_df['class'] == 'Instruction-edited'],
        orig_df[orig_df['class'] == 'Inpainting'],
        orig_df[orig_df['class'] == 'Inpainting + \n Refining'],
    ]:
        n_unique_images = len(df['image'].unique())

        print("Continuous")
        print(f"Pearson correlation: "
              f"{df.groupby('image')[['miou', 'pred']].mean().corr(method='pearson')['pred'][0]:0.4f}")
        print(f"Spearman correlation: "
              f"{df.groupby('image')[['miou', 'pred']].mean().corr(method='spearman')['pred'][0]:0.4f}")
        print(f"Kendall correlation: "
              f"{df.groupby('image')[['miou', 'pred']].mean().corr(method='kendall')['pred'][0]:0.4f}")

        print("Categorical")
        temp_df = df.groupby('image')[['true', 'pred']].mean() > 1 / 2
        print(f"Jaccard sim: "
              f"{jaccard_score(temp_df['true'], temp_df['pred']):0.4f}")
        print(f"Dice disim: "
              f"{dice(temp_df['true'], temp_df['pred']):0.4f}")

        agreement = (df.groupby(['image', 'true'])['pred'].mean().between(1 / 2, 1 / 2)).reset_index().groupby('image')[
            'pred'].max().sum()
        nv_agreement = (df.groupby(['image', 'true'])['pred'].mean() < 1 / 2).reset_index()['pred'].sum()
        v_agreement = (df.groupby(['image', 'true'])['pred'].mean() > 1 / 2).reset_index()['pred'].sum()
        print(f"Agreement with 50%: "
              f"{n_unique_images - agreement} ({(n_unique_images - agreement) / n_unique_images * 100:0.2f}%) - "
              f"{nv_agreement} ({nv_agreement / n_unique_images * 100:0.2f}%) - "
              f"{v_agreement} ({v_agreement / n_unique_images * 100:0.2f}%)")

        agreement = (df.groupby(['image', 'true'])['pred'].mean().between(1 / 3, 2 / 3)).reset_index().groupby('image')[
            'pred'].max().sum()
        nv_agreement = (df.groupby(['image', 'true'])['pred'].mean() < 1 / 3).reset_index()['pred'].sum()
        v_agreement = (df.groupby(['image', 'true'])['pred'].mean() > 2 / 3).reset_index()['pred'].sum()
        print(f"Agreement with 66%: "
              f"{n_unique_images - agreement} ({(n_unique_images - agreement) / n_unique_images * 100:0.2f}%) - "
              f"{nv_agreement} ({nv_agreement / n_unique_images * 100:0.2f}%) - "
              f"{v_agreement} ({v_agreement / n_unique_images * 100:0.2f}%)")

        temp_df = (df.groupby(['image', 'true', 'class'])['pred'].mean().between(1 / 2, 1 / 2)).reset_index().groupby(['image', 'class'])['pred'].mean()
        print(temp_df[temp_df == 1].index.tolist())


        temp_df = (df.groupby(['image', 'true'])['pred'].mean() > 1/2).reset_index()[
            ['true', 'pred']].value_counts().reset_index()
        TP = temp_df[(temp_df['true'] == True) * (temp_df['pred'] == True)]['count'].values[0]
        FN = temp_df[(temp_df['true'] == False) * (temp_df['pred'] == True)]['count'].values[0]
        temp_df = (df.groupby(['image', 'true'])['pred'].mean() < 1/2).reset_index()[
            ['true', 'pred']].value_counts().reset_index()
        x = temp_df[(temp_df['true'] == False) * (temp_df['pred'] == True)]['count']
        TN = x.values[0] if len(x) > 0 else 0
        x = temp_df[(temp_df['true'] == True) * (temp_df['pred'] == True)]['count']
        FP = x.values[0] if len(x) > 0 else 0
        # FP = temp_df[(temp_df['true'] == True) * (temp_df['pred'] == True)]['count']
        print(f"TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}")

        temp_df = (df.groupby(['image', 'true'])['pred'].mean() > 2/3).reset_index()[
            ['true', 'pred']].value_counts().reset_index()
        TP = temp_df[(temp_df['true'] == True) * (temp_df['pred'] == True)]['count'].values[0]
        FN = temp_df[(temp_df['true'] == False) * (temp_df['pred'] == True)]['count'].values[0]
        temp_df = (df.groupby(['image', 'true'])['pred'].mean() < 1/3).reset_index()[
            ['true', 'pred']].value_counts().reset_index()
        x = temp_df[(temp_df['true'] == False) * (temp_df['pred'] == True)]['count']
        TN = x.values[0] if len(x) > 0 else 0
        x = temp_df[(temp_df['true'] == True) * (temp_df['pred'] == True)]['count']
        FP = x.values[0] if len(x) > 0 else 0
        # FP = temp_df[(temp_df['true'] == True) * (temp_df['pred'] == True)]['count']
        print(f"TP: {TP}, FN: {FN}, TN: {TN}, FP: {FP}")

        print(df.groupby(['class', 'true'])['pred'].mean())
