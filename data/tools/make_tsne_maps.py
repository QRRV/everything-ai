# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/qmnist-tsne-maps.npz: t-SNE maps of the 1,000 test digits of data/qmnist-small.npz, for the lesson
ml-tsne. A map takes a while to make, so the lesson reads these instead of making four at every start; its worked
example makes one of its own, of fewer digits.

    python data/tools/make_tsne_maps.py

Each map is scikit-learn's TSNE(n_components=2, perplexity=p, random_state=0) of the test images divided by 255,
everything else at scikit-learn's defaults (init "pca", max_iter 1000), for every p in PERPLEXITIES. It is saved as
float32 under the name "p" + p, such as "p30", with the scikit-learn version under "sklearn". One more map, "p30-seed1",
is the one with perplexity 30 and random_state=1, to show that another seed, from the same start by PCA, draws nearly the same map. The result is written
to data/, this folder's parent."""
from pathlib import Path

import numpy as np

PERPLEXITIES = [2, 5, 30, 100]


def main() -> None:
    import sklearn
    from sklearn.manifold import TSNE
    data = Path(__file__).resolve().parents[1]
    with np.load(data / "qmnist-small.npz") as d:
        X = d["test_images"].astype(float) / 255
    maps = {f"p{p}": TSNE(n_components=2, perplexity=p, random_state=0).fit_transform(X).astype(np.float32)
            for p in PERPLEXITIES}
    maps["p30-seed1"] = TSNE(n_components=2, perplexity=30, random_state=1).fit_transform(X).astype(np.float32)
    np.savez_compressed(data / "qmnist-tsne-maps.npz", sklearn=np.array(sklearn.__version__), **maps)
    print("written", data / "qmnist-tsne-maps.npz", "with scikit-learn", sklearn.__version__)


if __name__ == "__main__":
    main()
