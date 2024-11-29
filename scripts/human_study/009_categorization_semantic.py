import matplotlib
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_score

from utils.path_utils import PROJECT_DIR
import statsmodels.api as sm
import pandas as pd

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

    def read_file(path):
        df = pd.read_csv(path, index_col=0)
        df['user'] = str(path.name).split('_')[1]
        return df

    # Read all dfs:
    df = pd.concat([read_file(file)
                    for file in
                    PROJECT_DIR.joinpath('scripts', 'human_study', 'data', 'custom').glob("semantic*.csv")])

    df['image'] = df['left_image'].map(lambda x: x.split("/")[1].replace('original', 'augmented'))
    df['class'] = df['image'].map(name_approach_map)
    df['class'] = df['class'].map(class_rename_dict)
    df['miou'] = df['image'].map(name_miou_map)
    df['true'] = df['miou'] >= 0.9
    df['pred'] = df['answer'] == 'yes'
    df = df[~df['class'].isna()]

    orig_df = df.copy(deep=True)
    result = {}

    for user in orig_df['user'].unique():

        df = orig_df[orig_df['user'] == user]

        for df in [
            df
        ]:

            n_unique_images = len(df['image'].unique())

            print("Continuous")
            temp_df = df.groupby('image')[['miou', 'pred']].mean()
            print(f"Pearson correlation: "
                  f"{temp_df.groupby('image')[['miou', 'pred']].mean().corr(method='pearson')['pred'][0]:0.4f}")
            print(f"Spearman correlation: "
                  f"{temp_df.groupby('image')[['miou', 'pred']].mean().corr(method='spearman')['pred'][0]:0.4f}")
            print(f"Kendall correlation: "
                  f"{temp_df.groupby('image')[['miou', 'pred']].mean().corr(method='kendall')['pred'][0]:0.4f}")

            print("Categorical")
            temp_df = df.groupby('image')[['true', 'pred']].mean() > 1 / 2
            print(f"Jaccard sim: "
                  f"{jaccard_score(temp_df['true'], temp_df['pred']):0.4f}")
            print(f"Dice disim: "
                  f"{dice(temp_df['true'], temp_df['pred']):0.4f}")

            print("Biserial Correlation")
            temp_df = df.groupby('image')[['true', 'pred', 'miou']].mean()
            temp_df['pred'] = temp_df['pred'] > 1 / 2
            print(f"Biserial: {scipy.stats.pointbiserialr(temp_df['miou'], temp_df['pred'])[0]:0.4f}")


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

            result[user] = float(FP/57)
            print(result.keys())
            print(result.values())

data = pd.read_csv(
    "/home/banana/Downloads/Participant Expertise and Image Validation Survey.csv (1)/Participant Expertise and Image Validation Survey.csv",
    header=1,
    names=['time', 'username', 'has_license', 'license_years', 'ads_knowledge', 'ads_expertise', 'has_vision_issue', 'vision_quality', 'ads_experience']
)

data = data.merge(pd.DataFrame(data={
        'username': result.keys(),
        'fnr': result.values(),
    }), on='username', how='left')

df = data
df['has_license'] = df['has_license'].map(lambda x: x == "Yes").astype(int)
df['ads_knowledge'] = df['ads_knowledge'].map(lambda x: x == "Yes").astype(int)
df['has_vision_issue'] = df['has_vision_issue'].map(lambda x: x == "Yes").astype(int)
df['ads_experience'] = df['ads_experience'].map(lambda x: x == "Yes").astype(int)

# for col in set(data.columns).difference(['time', 'username']):
X = df[list(set(data.columns).difference(['time', 'username', 'fnr']))]
y = df['fnr']
X = sm.add_constant(X)

model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(model.summary())