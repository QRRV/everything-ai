# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# A testbed for your agents: the Everything AI course put the drifting bandits of an earlier lesson here.
# Een testbed voor je agents: de Everything AI-course heeft hier de verschuivende bandits van een eerdere les gezet.
# It is your file: your run.py lets your agents play these bandits.
# Het is jouw bestand: je run.py laat je agents op deze bandits spelen.
import numpy as np


class DriftingBandit:
    # k machines; every mean starts around 0, with standard deviation 1
    # k automaten; elk gemiddelde begint rond 0, met standaarddeviatie 1
    def __init__(self, k, drift, rng):
        self.rng = rng
        self.drift = drift
        self.means = rng.normal(0.0, 1.0, k)

    def best(self):
        # the machine with the highest mean right now; only for measuring, the agent never sees it
        # de automaat met het hoogste gemiddelde op dit moment; alleen om te meten, de agent ziet dit nooit
        return int(np.argmax(self.means))

    def pull(self, action):
        # the reward of one pull: the mean of the machine plus noise with standard deviation 1;
        # then every mean takes a small random step, with standard deviation `drift`
        # de reward van één keer trekken: het gemiddelde van de automaat plus ruis met standaarddeviatie 1;
        # daarna zet elk gemiddelde een kleine willekeurige stap, met standaarddeviatie `drift`
        reward = float(self.means[action] + self.rng.normal())
        self.means = self.means + self.rng.normal(0.0, self.drift, len(self.means))
        return reward
