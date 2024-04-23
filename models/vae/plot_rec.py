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
from scipy.stats import gamma

# for domain in WEATHERS:

domain = "usa"

vae_filename = "vae_rec_9.csv"

path = RESULT_DIR.joinpath('ip2p', 'make_it_usa-2_0', 'before')

df = pd.read_csv(path.joinpath(vae_filename))

shape, loc, scale = gamma.fit(df['rec_loss'], floc=0)
x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
# gamma.ppf(c, shape, loc=loc, scale=scale)
plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label='nominal')


path = RESULT_DIR.joinpath('ip2p', f'make_it_{domain}-1_5', 'after')

df = pd.read_csv(path.joinpath(vae_filename))

shape, loc, scale = gamma.fit(df['rec_loss'], floc=0)
x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
# gamma.ppf(c, shape, loc=loc, scale=scale)
plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f'{domain}-1_5')




path = RESULT_DIR.joinpath('ip2p', f'make_it_{domain}-2_0', 'after')

df = pd.read_csv(path.joinpath(vae_filename))

shape, loc, scale = gamma.fit(df['rec_loss'], floc=0)
x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
# gamma.ppf(c, shape, loc=loc, scale=scale)
plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f'{domain}-2_0')




path = RESULT_DIR.joinpath('ip2p', f'make_it_{domain}-2_5', 'after')

df = pd.read_csv(path.joinpath(vae_filename))

shape, loc, scale = gamma.fit(df['rec_loss'], floc=0)
x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
# gamma.ppf(c, shape, loc=loc, scale=scale)
plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f'{domain}-2_5')




path = RESULT_DIR.joinpath('ip2p', f'make_it_{domain}-3_0', 'after')

df = pd.read_csv(path.joinpath(vae_filename))

shape, loc, scale = gamma.fit(df['rec_loss'], floc=0)
x = np.linspace(gamma.ppf(0.001, shape, loc=loc, scale=scale),
                gamma.ppf(0.999, shape, loc=loc, scale=scale), 1000)
# gamma.ppf(c, shape, loc=loc, scale=scale)
plt.plot(x, gamma.pdf(x, shape, loc=loc, scale=scale), lw=5, alpha=0.6, label=f'{domain}-3_0')



plt.legend()
plt.savefig(f"vae_rec_{domain}_9.png")
plt.close()