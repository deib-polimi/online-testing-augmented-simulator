import PIL.Image
import numpy as np
import torch
import torchmetrics
import tqdm

from utils.path_utils import RESULT_DIR

metric = torchmetrics.classification.BinaryJaccardIndex()
metric_values = []

folder = RESULT_DIR.joinpath("test", "before_segmentation")
for filepath in tqdm.tqdm([x for x in sorted(list(folder.iterdir())) if x.suffix == '.jpg']):
    filename = filepath.name

    ground_truth_filepath = RESULT_DIR.joinpath("test", "before_segmentation",
                                                f"semantic_segmentation_{filename.split('_')[-1]}")
    prediction_filepath = RESULT_DIR.joinpath("test", "segmentation_segformer",
                                              f"frame_{filename.split('_')[-1]}".replace("jpg", "png"))

    true = np.array(PIL.Image.open(ground_truth_filepath))
    true = np.all(true == (0, 0, 254), axis=-1)

    pred = np.array(PIL.Image.open(prediction_filepath))
    pred = pred == 255

    x = metric(torch.tensor(pred), torch.tensor(true))

    metric_values.append(x.item())

metric_values = np.array(metric_values)
print(metric_values.mean())
