# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# What the packed model of your project expects for a table of hours: the Everything AI course put
# this here for your behavioral tests. It reads the package in ops/model as the course reads it.
# Wat het verpakte model van je project verwacht voor een tabel met uren: de Everything AI-course
# zette dit hier neer voor je behavioral tests. Het leest het pakket in ops/model zoals de course
# het leest.
import json
import pathlib

import torch

from ops.demand import DemandNet, inputs
from ops.safetensors_io import load_file

FOLDER = pathlib.Path(__file__).parent / "model"


def load(folder=FOLDER):
    # the package that pack made: the network from config.json, its weights from model.safetensors
    # het pakket dat pack maakte: het netwerk uit config.json, de weights uit model.safetensors
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = DemandNet(config["hidden"])
    model.load_state_dict(load_file(folder / "model.safetensors"))
    model.eval()
    scale = {"mean": config["mean"], "std": config["std"]}

    def expect(hours):
        # what the model expects for each hour of the table, in bikes: a tensor of one number per hour
        # wat het model voor elk uur van de tabel verwacht, in fietsen: een tensor met één getal per uur
        with torch.no_grad():
            return model(inputs(hours, scale)).reshape(-1)

    return expect
