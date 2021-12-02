import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
from torchvision.utils import save_image
import torch.optim as optim

import config
import model
import utils

# generator model
gen = model.Generator(
    config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
scaler_gen = torch.cuda.amp.GradScaler()

utils.load_checkpoint(
    config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )

utils.generate_examples(gen, 8, truncation=0.7, n=200)
