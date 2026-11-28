# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# The PSI of the lesson When the world changes: drift, for the drift check of your service: the
# Everything AI course put it here. psi_of gets the values of a column themselves, not shares.
# De PSI uit de les Als de wereld verandert: drift, voor de driftcheck van je service: de Everything
# AI-course zette deze PSI hier neer. psi_of krijgt de waarden van een kolom zelf, geen aandelen.
import numpy as np

TENTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def edges(then):
    # ten bins that each hold about a tenth of then: minus infinity, the 0.1 to 0.9 quantiles, infinity
    # tien bins met elk ongeveer een tiende van then: min oneindig, de kwantielen 0,1 tot 0,9, oneindig
    return np.concatenate([[-np.inf], np.quantile(then, TENTHS), [np.inf]])


def shares(values, edges):
    # the share of the values in each bin: a value exactly on an edge goes into the bin above it
    # het aandeel van de waarden in elke bin: een waarde precies op een grens telt mee in de bin erboven
    out = []
    for i in range(len(edges) - 1):
        inside = values[values >= edges[i]]
        inside = inside[inside < edges[i + 1]]
        out.append(len(inside) / len(values))
    return np.array(out)


def psi_of(then, now):
    # the PSI of the values now against the values then, in the ten bins of then
    # de PSI van de waarden now tegenover de waarden then, in de tien bins van then
    e = edges(then)
    # an empty bin gets a tiny share instead of 0, so that b / a never divides by 0
    # een lege bin krijgt een heel klein aandeel in plaats van 0, zodat b / a nooit door 0 deelt
    a = np.maximum(shares(then, e), 0.0001)
    b = np.maximum(shares(now, e), 0.0001)
    return float(((b - a) * np.log(b / a)).sum())
