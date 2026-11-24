# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/xquad-topics.npz: the course's topic models of the 2,067 paragraphs of data/xquad-squad-nl.json.gz, one in
English and one in Dutch, for the lesson nlp-topic-models. Training one takes several seconds, so the lesson's
exercises and pictures read these; its demo trains the same model live, step by step.

    uv run python data/tools/make_xquad_topics.py

Each model is scikit-learn's LatentDirichletAllocation(n_components=12, random_state=0, max_iter=20), everything else at
scikit-learn's defaults (learning_method "batch", doc_topic_prior and topic_word_prior 1/12), fitted on the counts of
everything_ai.xquad.counts(lang). For each language the file holds "<lang>_words", the words of the columns;
"<lang>_weights", the model's components_ (12 rows of pseudo-counts, one per topic); and "<lang>_mix", the model's
transform of the counts (one row of 12 shares per paragraph), both as float32; and "sklearn", the version of
scikit-learn that made them. The result is written to data/, this folder's parent, with a fixed date in the zip, so that
a rerun with the same versions gives the same bytes."""
import io
import zipfile
from pathlib import Path

import numpy as np

TOPICS = 12
STEPS = 20


def main() -> None:
    import sklearn
    from sklearn.decomposition import LatentDirichletAllocation

    from everything_ai import xquad

    data = Path(__file__).resolve().parents[1]
    found = {"sklearn": np.array(sklearn.__version__)}
    for lang in xquad.LANGS:
        counts, words = xquad.counts(lang)
        model = LatentDirichletAllocation(n_components=TOPICS, random_state=0, max_iter=STEPS).fit(counts)
        found[f"{lang}_words"] = np.array(words)
        found[f"{lang}_weights"] = model.components_.astype(np.float32)
        found[f"{lang}_mix"] = model.transform(counts).astype(np.float32)
    with zipfile.ZipFile(data / "xquad-topics.npz", "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in found.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())
    print("written", data / "xquad-topics.npz", "with scikit-learn", sklearn.__version__)


if __name__ == "__main__":
    main()
