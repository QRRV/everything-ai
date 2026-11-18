# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT-0
# Your classifier for the audit: the Everything AI course put the one of its earlier lessons here.
# Je classifier voor de audit: de Everything AI-course heeft die van zijn eerdere lessen hier gezet.
# It is your file: change the columns, and the lesson audits the classifier you made.
# Het is jouw bestand: verander de kolommen, en de les doet de audit op de classifier die jij maakte.
import pandas as pd
from sklearn.linear_model import LogisticRegression

# the columns the classifier gets: numbers, and words that become a column of 0 and 1 per value
# it does not get sex, but relationship holds Husband and Wife, and it gets race
NUMBERS = ["age", "education_num", "hours_per_week"]
WORDS = ["workclass", "marital_status", "occupation", "relationship", "race"]


def features(people):
    # what the classifier gets, every column divided by its largest value; give it the whole table
    X = pd.get_dummies(people[NUMBERS + WORDS], columns=WORDS, dtype=float)
    return X / X.max()


def train(people):
    # the classifier, trained on the rows whose split is "train"; the label: earns more than 50,000 dollars
    X = features(people)
    train_rows = people["split"] == "train"
    model = LogisticRegression(max_iter=1000)
    model.fit(X[train_rows], people["income"][train_rows] == ">50K")
    return model


def scores(people):
    # for every person, the probability of a yes: that they earn more than 50,000 dollars
    return train(people).predict_proba(features(people))[:, 1]
