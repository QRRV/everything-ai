# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The orders of the British online shop of the course whose customer is known (data/online-retail-baskets.csv.gz,
see data/README.md), split by time as the lessons on recommending split them: every customer's last order is the
test, and the orders before it are all a recommender may use. Here are that split, the popularity baseline, and the
words of the product names that content-based filtering compares, so that every lesson of the module measures on
the same customers and the same hidden products."""
import functools

DATA = "data/online-retail-baskets.csv.gz"


@functools.lru_cache(maxsize=1)
def rows():
    """Every product of every order whose customer is known, with `last`: the number of the customer's last order.
    The invoices count up over time, so the largest is the last."""
    import pandas as pd
    from everything_ai import paths
    r = pd.read_csv(paths.ROOT / DATA).dropna(subset=["customer"])
    r["last"] = r.groupby("customer")["invoice"].transform("max")
    return r


@functools.lru_cache(maxsize=1)
def split() -> dict:
    """The size of the user-item matrix of all orders; then, for every customer with an order before their last, the
    set of products they bought before it (`seen`), and the products of the last order they had not bought before
    (`hidden`, only the customers with at least one: the test customers); and `counts`, how many customers bought
    each product before their last order, the most first."""
    r = rows()
    pairs = r.drop_duplicates(subset=["customer", "product"])
    train = r[r["invoice"] < r["last"]]
    test = r[r["invoice"] == r["last"]]
    seen = dict(train.groupby("customer")["product"].agg(set).items())
    last = test.groupby("customer")["product"].agg(set)
    hidden = {}
    for c, s in seen.items():
        new = {p for p in last[c] if p not in s}
        if new:
            hidden[c] = new
    counts = train.drop_duplicates(subset=["customer", "product"])["product"].value_counts()
    return {"customers": len(set(pairs["customer"])), "products": len(set(pairs["product"])), "pairs": len(pairs),
            "seen": seen, "hidden": hidden, "counts": counts}


def popular_for(customer: float, k: int) -> list:
    """The popularity baseline for one customer: the k most popular products they have not bought."""
    s = split()
    seen = s["seen"][customer]
    return [p for p in s["counts"].index[:k + len(seen)] if p not in seen][:k]


def found(recommend, k: int = 10) -> dict:
    """For every test customer, whether at least one of the k products `recommend(customer, k)` gives is hidden."""
    s = split()
    return {c: any(p in h for p in recommend(c, k)) for c, h in s["hidden"].items()}


# --- the words of the product names ----------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def words() -> dict:
    """Every word of the names of the products in `counts`, as `name.split()` cuts them, with its column: in the order
    the words first turn up, going down `counts` from the most popular product."""
    out = {}
    for name in split()["counts"].index:
        for w in name.split():
            if w not in out:
                out[w] = len(out)
    return out


@functools.lru_cache(maxsize=1)
def name_vectors():
    """A row for every product of `counts`, in its order, and a column for every word: 1 where the word is in the name."""
    import numpy as np
    w = words()
    names = list(split()["counts"].index)
    F = np.zeros((len(names), len(w)))
    for i, name in enumerate(names):
        for word in name.split():
            F[i, w[word]] = 1
    F.flags.writeable = False
    return F


def profile_of(customer: float):
    """The user profile: the rows of everything the customer bought before their last order, added up."""
    import numpy as np
    s = split()
    at = {p: i for i, p in enumerate(s["counts"].index)}
    F = name_vectors()
    out = np.zeros(F.shape[1])
    for p in s["seen"][customer]:
        out = out + F[at[p]]
    return out


def content_scores(customer: float):
    """The cosine similarity of every product's row with the customer's profile."""
    import numpy as np
    F, p = name_vectors(), profile_of(customer)
    return F @ p / (np.linalg.norm(F, axis=1) * np.linalg.norm(p))


def content_for(customer: float, k: int) -> list:
    """Content-based filtering for one customer: the k products they have not bought whose rows are closest to their
    profile. Products with the same score keep the order of `counts`, the most popular first."""
    import numpy as np
    s = split()
    names = list(s["counts"].index)
    score = content_scores(customer)
    seen = s["seen"][customer]
    out = []
    for i in np.argsort(-score, kind="stable"):
        if names[i] not in seen:
            out.append(names[i])
            if len(out) == k:
                break
    return out


@functools.lru_cache(maxsize=None)
def content_found(k: int = 10) -> dict:
    """`found(content_for, k)`, the same for every customer, but with the scores of 500 customers in one product of
    matrices: a few seconds instead of a minute. The scores are sums of whole numbers, so both ways give the same."""
    import numpy as np
    s = split()
    names = list(s["counts"].index)
    at = {p: i for i, p in enumerate(names)}
    F = name_vectors()
    norms = np.linalg.norm(F, axis=1)
    customers = list(s["hidden"])
    out = {}
    for start in range(0, len(customers), 500):
        chunk = customers[start:start + 500]
        P = np.zeros((len(chunk), F.shape[1]))
        for r, c in enumerate(chunk):
            P[r] = F[[at[p] for p in s["seen"][c]]].sum(axis=0)
        S = P @ F.T / (np.linalg.norm(P, axis=1)[:, None] * norms[None, :])
        for r, c in enumerate(chunk):
            row = S[r].copy()
            row[[at[p] for p in s["seen"][c]]] = -np.inf
            out[c] = any(names[i] in s["hidden"][c] for i in np.argsort(-row, kind="stable")[:k])
    return out


# --- who bought what: collaborative filtering ------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def matrix():
    """The user-item matrix of the orders before every customer's last: a row for every customer of `seen`, in its
    order, a column for every product of `counts`, in its order, and a 1 where the customer bought the product."""
    import numpy as np
    s = split()
    at = {p: i for i, p in enumerate(s["counts"].index)}
    R = np.zeros((len(s["seen"]), len(at)))
    for r, c in enumerate(s["seen"]):
        for p in s["seen"][c]:
            R[r, at[p]] = 1
    R.flags.writeable = False
    return R


@functools.lru_cache(maxsize=1)
def item_similarities():
    """The cosine similarity of every column of `matrix()` with every other: how alike two products are by who bought
    them. The customers who bought both, divided by the length of each column."""
    import numpy as np
    R = matrix()
    lengths = np.linalg.norm(R, axis=0)
    S = (R.T @ R) / (lengths[:, None] * lengths[None, :])
    S.flags.writeable = False
    return S


def bought_columns(customer: float) -> list:
    """The columns of the products a customer bought before their last order, smallest first, so that sums over them
    always add up in the same order."""
    at = {p: i for i, p in enumerate(split()["counts"].index)}
    return sorted(at[p] for p in split()["seen"][customer])


def collaborative_scores(customer: float):
    """Item-item collaborative filtering for one customer: for every product, its similarities with the products they
    bought, added up."""
    return item_similarities()[bought_columns(customer)].sum(axis=0)


def collaborative_for(customer: float, k: int) -> list:
    """The k products a customer has not bought with the highest collaborative scores; equal scores in the order of
    `counts`, the most popular first."""
    import numpy as np
    s = split()
    names = list(s["counts"].index)
    seen = s["seen"][customer]
    score = collaborative_scores(customer)
    out = []
    for i in np.argsort(-score, kind="stable"):
        if names[i] not in seen:
            out.append(names[i])
            if len(out) == k:
                break
    return out


@functools.lru_cache(maxsize=None)
def collaborative_found(k: int = 10) -> dict:
    """`found(collaborative_for, k)`: for every test customer, whether one of their k products is hidden."""
    return found(collaborative_for, k)
