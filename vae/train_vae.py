import torch

from data.dataset import ImageDataset
from utils.conf import RESULT_DIR

if __name__ == '__main__':

    dataset = ImageDataset(path=RESULT_DIR.joinpath('ip2p', 'make_it_cloudy-2_5', 'before'))

    # dataloader = torch.utils.data.DataLoader()

