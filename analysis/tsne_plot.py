import numpy as np
import pandas as pd
import torch
from torch import flatten
from torch.utils.data import DataLoader
from cuml.manifold import TSNE
from ads.model import UdacityDrivingModel
from data.dataset import ImageDataset
from utils.path_utils import PROJECT_DIR, RESULT_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from augment.domain import WEATHERS, COUNTRIES, LOCATIONS, TIMES, SEASONS

if __name__ == '__main__':

    imgs_per_domain = 20
    batch_size = 10
    device = "cuda:1"

    path = RESULT_DIR.joinpath('ip2p', 'make_it_cloudy-2_0', 'before')

    checkpoint = PROJECT_DIR.joinpath("lake_sunny_day_60_0.ckpt")
    model = UdacityDrivingModel("nvidia_dave", (3, 160, 320))
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict'])
    model = model.model[:-15]
    model = model.to(device)
    model.eval()

    embeddings = []
    labels = []
    df = pd.DataFrame()

    dataset = ImageDataset(path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=2,
        num_workers=16,
    )

    with torch.no_grad():
        for i, imgs in enumerate(dataloader):

            if i >= imgs_per_domain:
                break

            imgs = imgs.to(device)
            embeddings += flatten(model(imgs), start_dim=1).detach().cpu()

    embeddings = torch.vstack(embeddings).numpy()
    labels += ["nominal"] * len(embeddings)

    # domains = [x for x in sorted(list(RESULT_DIR.joinpath('ip2p').iterdir())) if "1_5" in x.name][:4]
    # domains = [RESULT_DIR.joinpath('ip2p', f'make_it_{x}-1_5') for x in WEATHERS]
    domains = [RESULT_DIR.joinpath('ip2p', f'make_it_rainy-{x}') for x in ['1_5', '2_0', '2_5', '3_0']]

    for n, directory in enumerate(domains):
        domain = directory.name
        path = directory.joinpath('after')
        if not directory.exists():
            continue

        dataset = ImageDataset(path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            prefetch_factor=2,
            num_workers=16,
        )

        e = []
        with torch.no_grad():
            for i, imgs in enumerate(dataloader):

                if i >= imgs_per_domain:
                    break

                imgs = imgs.to(device)
                e += flatten(model(imgs), start_dim=1).detach().cpu()

        embeddings = np.concatenate([embeddings, torch.vstack(e).numpy()])
        labels += [domain] * len(e)

    tsne_2D = TSNE(n_components=2, perplexity=5, learning_rate=20, n_iter=1000, verbose=1).fit_transform(embeddings)



    # for i in range(len(labels)//len(e) - 1):
    #     x, y = tsne_2D.T
    #     fig, ax = plt.subplots(figsize=(12, 12))
    #     ax.scatter(x[i * len(e): (i+1) * len(e)], y[i * len(e): (i+1) * len(e)], c=labels[i * len(e): (i+1) * len(e)], cmap=plt.cm.plasma)
    x, y = tsne_2D.T
    df = pd.DataFrame(
        {
            'domain': labels,
            'x': x,
            'y': y,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    # ax.scatter(x, y, c=labels, cmap=plt.cm.plasma)
    plot = sns.scatterplot(data=df, x="x", y="y", hue="domain")
    plot.get_figure().savefig("tsne.png")
    # for i, directory in enumerate(RESULT_DIR.joinpath('ip2p').iterdir()):
    #
    #     path = directory.joinpath('after')
    #
    #     dataset = ImageDataset(path)
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         prefetch_factor=2,
    #         num_workers=16,
    #     )
    #
    #     embeddings = []
    #
    #     with torch.no_grad():
    #         for i, imgs in enumerate(dataloader):
    #
    #             if i >= imgs_per_domain:
    #                 break
    #
    #             imgs = imgs.to(device)
    #             embeddings += model(imgs).detach().cpu()
    #
    #     embeddings = torch.vstack(embeddings).numpy()
    #
    #     tsne_2D = TSNE(n_components=2, perplexity=15, learning_rate=10, verbose=1).fit_transform(embeddings)
    #
    #     x, y = tsne_2D.T
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     ax.scatter(x, y, label="wow", c=[i] * len(x), cmap=plt.cm.plasma)

# plt.show()