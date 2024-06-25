#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import os
import torch
import random
import numpy as np


def set_seed(
        seed: int
        ) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)