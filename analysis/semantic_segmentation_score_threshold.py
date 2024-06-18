import pathlib

import PIL.Image
import pandas as pd
import torch
import torchmetrics
import torchvision.transforms
import tqdm
import seaborn
import matplotlib.pyplot as plt
from domains.prompt import ALL_PROMPTS
from models.augmentation.stable_diffusion_inpainting import StableDiffusionInpainting
from models.segmentation.unet_attention import SegmentationUnet
from utils.conf import DEFAULT_DEVICE
from utils.path_utils import PROJECT_DIR, MODEL_DIR

if __name__ == '__main__':

    # Step 0. Setup and config information
    # Handcrafted dataset, we manually picked and classified different types of roads
    dataset_folder = PROJECT_DIR.joinpath("sss_dataset")
    image_paths = [x for x in list(dataset_folder.glob('**/*.jpg'))
                   if "augmix" not in str(x) and "inpainting" not in str(x)]
    # Data augmentations settings
    aug_repetitions = 10

    # Step 1. Data management and augmentation
    # Apply Augmix transformation
    augmix = torchvision.transforms.AugMix()
    for img_path in image_paths:
        for i in range(aug_repetitions):
            output_file = pathlib.Path(str(img_path).replace('.jpg', f'_augmix_{i}.jpg'))
            if not output_file.exists():
                img = PIL.Image.open(img_path)
                augmented_img = augmix(img)
                augmented_img.save(output_file)
    # Apply Inpainting transformation
    model = StableDiffusionInpainting(prompt="", guidance=10)
    for i, prompt in enumerate(ALL_PROMPTS * aug_repetitions):
        img_path = image_paths[i % len(image_paths)]
        output_file = pathlib.Path(str(img_path).replace('.jpg', f'_inpainting_{prompt}.jpg'))
        if not output_file.exists():
            img = PIL.Image.open(img_path)
            model.prompt = prompt
            augmented_img = model(img)
            torchvision.utils.save_image(augmented_img, output_file)

    # Step 2. Compute segmentation
    image_paths = [x for x in list(dataset_folder.glob('**/image*.jpg'))]
    for img_path in image_paths:
        output_file = pathlib.Path(str(img_path).replace('image', 'new_segmentation').replace("jpg", "png"))
        if not output_file.exists():
            img = PIL.Image.open(img_path)
            mask = model.segmentation_mask_model(
                torchvision.transforms.functional.to_tensor(img).to(DEFAULT_DEVICE).unsqueeze(0))
            torchvision.utils.save_image(mask, output_file)

    # Step 3. Compute mIoU
    miou_results = []
    imgs1 = []
    imgs2 = []
    output_file = dataset_folder.joinpath("new_miou.csv")
    # if not output_file.exists():
    seg_paths = [x for x in list(dataset_folder.glob('**/new_segmentation*.png'))]
    seg_imgs = [
        torch.round(torchvision.transforms.functional.to_tensor(PIL.Image.open(seg_path)).to(DEFAULT_DEVICE)) for
        seg_path in seg_paths]
    miou_metric = torchmetrics.classification.BinaryJaccardIndex().to(DEFAULT_DEVICE)
    for i, seg_img1 in tqdm.tqdm(enumerate(seg_imgs)):
        for j, seg_img2 in enumerate(seg_imgs):
            imgs1.append(seg_paths[i])
            imgs2.append(seg_paths[j])
            # miou_results += [miou_metric(seg_img1[:, 20:120, :], seg_img2[:, 20:120, :]).item()]
            miou_results += [miou_metric(seg_img1, seg_img2).item()]
    df = pd.DataFrame({
        'img1': imgs1,
        'img2': imgs2,
        'miou': miou_results,
    })
    df['class1'] = df['img1'].map(lambda x: x.parent.name)
    df['class2'] = df['img2'].map(lambda x: x.parent.name)
    df.to_csv(output_file)

    # Step 4. Find threshold
    df = pd.read_csv(dataset_folder.joinpath("new_miou.csv"), index_col=0)
    direction_map = {
        'sharp_left_turn': 'left',
        'sharp_right_turn': 'right',
        'straightforward': 'center',
    }
    analysis_df = df.copy(deep=True)
    analysis_df = analysis_df[analysis_df['img1'].map(lambda x: "inpainting" not in x)]
    analysis_df = analysis_df[analysis_df['img2'].map(lambda x: "inpainting" not in x)]
    analysis_df['class1'] = analysis_df['class1'].map(direction_map)
    analysis_df['class2'] = analysis_df['class2'].map(direction_map)
    analysis_df = analysis_df[~analysis_df['class1'].isna()]
    analysis_df = analysis_df[~analysis_df['class2'].isna()]

    order = ["left", "center", "right"]

    a = analysis_df[analysis_df['class1'] == "left"]
    seaborn.displot(a, x='miou', hue="class2", kind="kde", height=1.5, aspect=3.33)
    plt.xlim(0, 1)
    plt.savefig("left.png")

    a = analysis_df[analysis_df['class1'] == "right"]
    seaborn.displot(a, x='miou', hue="class2", kind="kde", height=1.5, aspect=3.33)
    plt.xlim(0, 1)
    plt.savefig("right.png")

    a = analysis_df[analysis_df['class1'] == "center"]
    seaborn.displot(a, x='miou', hue="class2", kind="kde", height=1.5, aspect=3.33)
    plt.xlim(0, 1)
    plt.savefig("center.png")

    g = seaborn.FacetGrid(analysis_df, col="class1", row="class2", col_order=order, row_order=order,
                          height=3, aspect=2,
                          )
    g.map(seaborn.histplot, "miou", bins=30, binrange=(0, 1))
    plt.savefig("histplot.png")

    g = seaborn.FacetGrid(analysis_df, col="class1", row="class2", col_order=order, row_order=order,
                          height=3, aspect=2,
                          )
    g.map(seaborn.kdeplot, "miou")
    plt.savefig("displot.png")

    analysis_df.groupby(["class1", "class2"])['miou'].describe(
        percentiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.85, 0.9, 0.95,
                     0.975, 0.99]
    ).to_csv("data.csv")

    print(
        analysis_df[analysis_df['miou'] < 0.75].groupby(["class1", "class2"]).count() / analysis_df.groupby(
            ["class1", "class2"]).count())
    print(
        analysis_df[analysis_df['miou'] < 0.8].groupby(["class1", "class2"]).count() / analysis_df.groupby(
            ["class1", "class2"]).count())
    print(
        analysis_df[analysis_df['miou'] < 0.85].groupby(["class1", "class2"]).count() / analysis_df.groupby(
            ["class1", "class2"]).count())
    print(
        analysis_df[analysis_df['miou'] < 0.9].groupby(["class1", "class2"]).count() / analysis_df.groupby(
            ["class1", "class2"]).count())


    analysis_df = df.copy(deep=True)
    analysis_df = analysis_df[analysis_df['img1'].map(lambda x: "inpainting" in x)]
    analysis_df = analysis_df[analysis_df['img2'].map(lambda x: "inpainting" not in x and "augmix" not in x)]

    analysis_df['_img1'] = analysis_df['img1'].map(lambda x: pathlib.Path(x).stem.split('_')[0])
    analysis_df['_img2'] = analysis_df['img2'].map(lambda x: pathlib.Path(x.replace('image', 'segmentation').replace("jpg", "png")).stem)
    filter_array = [True if y in x else False for x,y in zip(analysis_df['img1'], analysis_df['_img2'])]
    analysis_df = analysis_df[filter_array]
    print(analysis_df.describe(
        percentiles=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.85, 0.9, 0.95,
                     0.975, 0.99]
    ))