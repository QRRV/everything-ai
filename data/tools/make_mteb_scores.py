# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/mteb-eng-v2.tsv: the scores of 15 embedding models on the 41 tasks of the English MTEB benchmark of the
new leaderboard, MTEB(eng, v2), with each model's number of parameters. Lesson nlp-mteb reads a leaderboard with them.

MTEB (Muennighoff, Tazi, Magne and Reimers, EACL 2023) and MMTEB (Enevoldsen and others, ICLR 2025) keep every result
in the repository embeddings-benchmark/results: one JSON file per model and task, with the main score of the task
(accuracy for classification, v-measure for clustering, average precision for pair classification, MAP for reranking,
nDCG@10 for retrieval, Spearman correlation for STS and summarization), and a model_meta.json per model. The
repository is CC0 1.0 ("free to redistribute, modify and adapt"). This script reads the files of 15 models at a fixed
commit, straight from GitHub, without keeping them:

    uv run python data/tools/make_mteb_scores.py

What it keeps: for each model and task, the main score of the test split, of the English subset where a task has
several ("en", or "en-en" for STS17), times 100, as the leaderboard shows it; and the model's n_parameters. The 41 tasks
and their types are those of MTEB(eng, v2) in the mteb package (mteb/benchmarks/benchmarks/benchmarks.py, MTEB_EN, at
commit 86d02b557aad1edfd2f10324a68273999155b8b6). Models whose results on these tasks come in part from their own
makers (an external/ folder in the repository) are left out, so every score here was computed with the mteb package.

What the file holds: a header line `model<TAB>parameters<TAB>type<TAB>task<TAB>score`, then one line per model and task,
the models in the order of MODELS and the tasks in the order of TASKS; `score` has four decimals.
"""
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "2aa494226cb43a73ac028d865d569e38cb7151e2"   # embeddings-benchmark/results, 2026-11-23
BASE = f"https://raw.githubusercontent.com/embeddings-benchmark/results/{COMMIT}/results"
OUT = Path(__file__).resolve().parents[1] / "mteb-eng-v2.tsv"

MODELS = [   # the folder of each model in the repository, and the revision its results were computed with
    ("sentence-transformers/all-MiniLM-L6-v2", "8b3219a92973c328a8e22fadcfa821b5dc75636a"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "bf3bf13ab40c3157080a7ab344c831b9ad18b5eb"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "79f2382ceacceacdf38563d7c5d16b9ff8d725d6"),
    ("sentence-transformers/LaBSE", "e34fab64a3011d2176c99545a93d5cbddc9a91b7"),
    ("intfloat/multilingual-e5-small", "fd1525a9fd15316a2d503bf26ab031a61d056e98"),
    ("intfloat/multilingual-e5-base", "d13f1b27baf31030b7fd040960d60d909913633f"),
    ("intfloat/multilingual-e5-large-instruct", "baa7be480a7de1539afce709c8f13f833a510e0a"),
    ("ibm-granite/granite-embedding-30m-english", "eddbb57470f896b5f8e2bfcb823d8f0e2d2024a5"),
    ("ibm-granite/granite-embedding-125m-english", "e48d3a5b47eaa18e3fe07d4676e187fd80f32730"),
    ("ibm-granite/granite-embedding-107m-multilingual", "47db56afe692f731540413c67dd818ff492277e7"),
    ("ibm-granite/granite-embedding-278m-multilingual", "84e3546b88b0cb69f8078608a1df558020bcbf1f"),
    ("Qwen/Qwen3-Embedding-0.6B", "b22da495047858cce924d27d76261e96be6febc0"),
    ("Qwen/Qwen3-Embedding-8B", "4e423935c619ae4df87b646a3ce949610c66241c"),
    ("intfloat/e5-mistral-7b-instruct", "07163b72af1488142a360786df853f237b1a3ca1"),
    ("GritLM/GritLM-7B", "13f00a0e36500c80ce12870ea513846a066004af"),
]

TASKS = {   # the 41 tasks of MTEB(eng, v2), by type
    "Classification": ["AmazonCounterfactualClassification", "Banking77Classification", "ImdbClassification",
                       "MTOPDomainClassification", "MassiveIntentClassification", "MassiveScenarioClassification",
                       "ToxicConversationsClassification", "TweetSentimentExtractionClassification"],
    "Clustering": ["ArXivHierarchicalClusteringP2P", "ArXivHierarchicalClusteringS2S", "BiorxivClusteringP2P.v2",
                   "MedrxivClusteringP2P.v2", "MedrxivClusteringS2S.v2", "StackExchangeClustering.v2",
                   "StackExchangeClusteringP2P.v2", "TwentyNewsgroupsClustering.v2"],
    "PairClassification": ["SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus"],
    "Reranking": ["AskUbuntuDupQuestions", "MindSmallReranking"],
    "Retrieval": ["ArguAna", "CQADupstackGamingRetrieval", "CQADupstackUnixRetrieval", "ClimateFEVERHardNegatives",
                  "FEVERHardNegatives", "FiQA2018", "HotpotQAHardNegatives", "SCIDOCS", "TRECCOVID",
                  "Touche2020Retrieval.v3"],
    "STS": ["BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS17", "STS22.v2", "STSBenchmark"],
    "Summarization": ["SummEvalSummarization.v2"],
}
SUBSET = {"AmazonCounterfactualClassification": "en", "MTOPDomainClassification": "en",
          "MassiveIntentClassification": "en", "MassiveScenarioClassification": "en", "STS17": "en-en", "STS22.v2": "en"}


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    lines = ["model\tparameters\ttype\ttask\tscore"]
    for model, revision in MODELS:
        folder = f"{BASE}/{model.replace('/', '__')}/{revision}"
        parameters = int(fetch(f"{folder}/model_meta.json")["n_parameters"])
        for kind, tasks in TASKS.items():
            for task in tasks:
                found = [s for s in fetch(f"{folder}/{task}.json")["scores"]["test"]
                         if s["hf_subset"] == SUBSET.get(task, "default")]
                if len(found) != 1:
                    sys.exit(f"{model} {task}: {len(found)} scores for the subset")
                lines.append(f"{model}\t{parameters}\t{kind}\t{task}\t{100 * found[0]['main_score']:.4f}")
        print(model, file=sys.stderr)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT.name}: {len(lines) - 1} scores of {len(MODELS)} models", file=sys.stderr)


if __name__ == "__main__":
    main()
