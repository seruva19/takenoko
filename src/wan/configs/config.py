# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from wan.configs.wan_t2v_14B import t2v_14B
from wan.configs.wan_t2v_A14B import t2v_A14B

WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-A14B": t2v_A14B,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
    "704*1280": 704 * 1280,
    "1280*704": 1280 * 704,
}

SUPPORTED_SIZES = {
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
}
