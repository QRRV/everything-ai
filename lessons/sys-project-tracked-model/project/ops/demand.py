# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# Your demand model: the Everything AI course put the hours, the inputs and the network of your
# project here. It is your file: your train.py and your pack.py use it, and so do the next steps.
# Je model van de vraag: de Everything AI-course zette hier de uren, de inputs en het netwerk van
# je project neer. Het is jouw bestand: je train.py en je pack.py gebruiken het, en de volgende
# stappen ook.
import pandas as pd
import torch
from torch import nn

from everything_ai import paths

DATA = paths.ROOT / "data" / "bike-sharing-hour.csv.gz"
FEATURES = ["weekday", "workingday", "holiday", "weathersit", "temp", "atemp", "hum", "windspeed",
            "season", "mnth"]


def read_hours(year=2011):
    # every hour of one year, with the column day: 0 for 1 January, 364 for 31 December
    # elk uur van één jaar, met de kolom day: 0 voor 1 januari, 364 voor 31 december
    hours = pd.read_csv(DATA)
    hours = hours[hours["yr"] == year - 2011].copy()
    hours["day"] = (pd.to_datetime(hours["dteday"]) - pd.Timestamp(year, 1, 1)).dt.days
    return hours.reset_index(drop=True)


def scaling(train):
    # the mean and the standard deviation of every feature, over the training hours only
    # het gemiddelde en de standaardafwijking van elke feature, alleen over de trainingsuren
    return {"mean": [round(float(train[f].mean()), 6) for f in FEATURES],
            "std": [round(float(train[f].std()), 6) for f in FEATURES]}


def inputs(hours, scale):
    # a row per hour: the hour as 24 one-hot columns, then every feature standardized
    # een rij per uur: het uur als 24 one-hot-kolommen, dan elke feature gestandaardiseerd
    hour = nn.functional.one_hot(torch.tensor(hours["hr"].to_numpy()), 24).float()
    values = torch.tensor(hours[FEATURES].to_numpy(dtype="float32"))
    scaled = (values - torch.tensor(scale["mean"])) / torch.tensor(scale["std"])
    return torch.cat([hour, scaled], dim=1)


def targets(hours):
    # the number of bikes rented in each hour, as a column
    # het aantal gehuurde fietsen in elk uur, als kolom
    return torch.tensor(hours["cnt"].to_numpy(dtype="float32")).reshape(-1, 1)


class DemandNet(nn.Module):
    # 34 inputs, one layer of `hidden` neurons with ReLU, and one output: the bikes in the hour
    # 34 inputs, één layer van `hidden` neuronen met ReLU, en één output: de fietsen in het uur
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(24 + len(FEATURES), hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def train_epoch(model, opt, x, y, batch):
    # one epoch: every training hour once, in a new random order, in mini-batches of `batch` hours
    # één epoch: elk trainingsuur één keer, in een nieuwe willekeurige volgorde, per `batch` uren
    order = torch.randperm(len(x))
    for i in range(0, len(x), batch):
        part = order[i:i + batch]
        loss = ((model(x[part]) - y[part]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
