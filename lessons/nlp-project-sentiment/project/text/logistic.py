# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# Logistic regression for your text classifier: the Everything AI course put the functions of an earlier lesson here.
# Logistische regressie voor je tekstclassifier: de Everything AI-course heeft hier de functies van een eerdere les gezet.
# It is your file: your train.py and judge.py use it.
# Het is jouw bestand: je train.py en judge.py gebruiken het.
import numpy as np


def probability(row, weights, bias):
    # the probability of label 1, spam, for one row: the sigmoid of the bias plus every value times its weight
    z = bias
    for word in row:
        z = z + row[word] * weights[word]
    return 1 / (1 + np.exp(-z))


def gradients(examples, weights, bias):
    # the slopes of the mean log loss of the examples, for every weight and for the bias
    n = len(examples)
    grad = {word: 0.0 for word in weights}
    grad_bias = 0.0
    for row, label in examples:
        error = probability(row, weights, bias) - label
        for word in row:
            grad[word] = grad[word] + error * row[word] / n
        grad_bias = grad_bias + error / n
    return grad, grad_bias
