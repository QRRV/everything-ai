# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The course's own models on Hugging Face, the model hub the course uses. A file is downloaded once from its fixed
address, without an account, into work/cache/hub/, and only kept when its SHA-256 is that of the course's own copy
in data/ (data/README.md says how that copy was made). Without a connection, or when the hub's file is not that
file, the course's copy is used: a lesson works offline and gives the same numbers everywhere. After one failed try,
the same Python process does not wait for that repository again. Set EAI_OFFLINE to never try the hub, as the
tests do.

Other models, too large for the course to ship, are downloaded from a repository on the hub (PINNED): once,
when a learner asks for it, from a fixed commit, and kept only when the SHA-256 is the one the course pinned. After
that the lesson works offline. Of a large file, a lesson can pin only its first bytes (`first`), such as the
beginning of a compressed shard of web pages: a learner then downloads only those. A file that is not on the hub has
its own address (`url`), at its source, and is pinned in the same way."""
import hashlib
import http.client
import os
import tempfile
import urllib.request
from pathlib import Path

from everything_ai import paths

HUB = "https://huggingface.co"
TIMEOUT = 10   # seconds to wait for the hub before the course's copy is used

# Per repository on the hub: each file and the course's own copy of it.
REPOS = {
    "QRRV/everything-ai-qmnist-cnn": {
        "model.pt": paths.ROOT / "data" / "course-qmnist-cnn.pt",
        "README.md": paths.ROOT / "data" / "course-qmnist-cnn.md",
    },
}


# Per file the course downloads from its own repository: the commit it comes from, its SHA-256 and its size in bytes.
PINNED = {
    ("openai-community/gpt2", "model.safetensors"): {
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "sha256": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
        "size": 548_105_171,
    },
    # Qwen3-0.6B-Base, an open model, and its tokenizer: llm-same-on-open-model (course/route/bronnen.yaml)
    ("Qwen/Qwen3-0.6B-Base", "model.safetensors"): {
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "sha256": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        "size": 1_192_135_096,
    },
    ("Qwen/Qwen3-0.6B-Base", "vocab.json"): {
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "size": 2_776_833,
    },
    ("Qwen/Qwen3-0.6B-Base", "tokenizer_config.json"): {
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "sha256": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
        "size": 9_678,
    },
    # Qwen3-0.6B, the same model after post-training, which follows the chat template and calls tools: the lessons
    # on prompts and agents and the domain bouwen (course/route/bronnen.yaml); its tokenizer is the Base's above
    ("Qwen/Qwen3-0.6B", "model.safetensors"): {
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "size": 1_503_300_328,
    },
    # the first 20 MB of a shard of C4 before its cleaning: raw web pages for llm-quality-filter (course/route/bronnen.yaml)
    ("datasets/allenai/c4", "en.noclean/c4-train.00000-of-07168.json.gz"): {
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "sha256": "2b74106294af67067b72ecd4a4075e6e386382a546bbf7f93960b5f90b7e747a",
        "size": 20_000_000,
        "first": True,
    },
    # the first 200 MB of the training stories of TinyStories: the data of llm-pretrain-capstone (course/route/bronnen.yaml)
    ("datasets/roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt"): {
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "sha256": "d4ab5db7423aee03fecad9dc029119c540259518dd73858bb58ef640c05cf160",
        "size": 200_000_000,
        "first": True,
    },
    # ResNet-18 with torchvision's own weights, trained on ImageNet, in timm's copy: cv-try-classifier and the lessons
    # after it (course/route/bronnen.yaml, model-resnet18-imagenet)
    ("timm/resnet18.tv_in1k", "model.safetensors"): {
        "revision": "bbd144b3e5565108aad885f145491d11bc6ce807",
        "sha256": "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81",
        "size": 46_807_446,
    },
    # DINOv2-small, a vision foundation model of Meta (Apache-2.0): cv-pretrained-features and the lessons after it
    # (course/route/bronnen.yaml, model-dinov2-small)
    ("facebook/dinov2-small", "model.safetensors"): {
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        "size": 88_249_960,
    },
    # DETR with ResNet-50, a detector of Meta trained on COCO (Apache-2.0): cv-pretrained-detector and the lessons after
    # it (course/route/bronnen.yaml, model-detr-resnet50)
    ("facebook/detr-resnet-50", "model.safetensors"): {
        "revision": "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b",
        "sha256": "830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81",
        "size": 166_587_896,
    },
    # SAM 2.1 tiny, a model of Meta that draws the mask of what you point at, trained on SA-1B and SA-V (Apache-2.0):
    # cv-sam (course/route/bronnen.yaml, model-sam2-1-hiera-tiny)
    ("facebook/sam2.1-hiera-tiny", "model.safetensors"): {
        "revision": "de431c4043854a71d8101e17995dfe596bf101a5",
        "sha256": "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7",
        "size": 155_908_064,
    },
    # Granite Embedding 107M multilingual, a text embedding model of IBM (Apache-2.0), and its tokenizer: everything_ai.embed
    ("ibm-granite/granite-embedding-107m-multilingual", "model.safetensors"): {
        "revision": "d6cffd338414d6a1c1f5decfad5fec62eebc90d5",
        "sha256": "ae6f20cc5dc433041118c900b652d87532e1335dade78c37765f0b94d1161283",
        "size": 214_000_232,
    },
    ("ibm-granite/granite-embedding-107m-multilingual", "tokenizer.json"): {
        "revision": "d6cffd338414d6a1c1f5decfad5fec62eebc90d5",
        "sha256": "7fc9e475f2ac473f070052d92b0c46ec2b91d9cd672935feb6fd837a65d33f78",
        "size": 9_081_382,
    },
    # Caltech-101 (CC BY 4.0), at its source, CaltechDATA (doi:10.22002/D1.20086): the photo set of the domain computer vision
    # (course/route/bronnen.yaml, caltech-101-download-only; everything_ai.caltech unpacks it). CaltechDATA gives the
    # same file's MD5, 3138e1922a9193bfa496528edbbc45d0.
    ("caltechdata/mzrjq-6wc02", "caltech-101.zip"): {
        "url": "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
        "revision": "v1",
        "sha256": "331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91",
        "size": 137_414_764,
    },
    # CIFAR-10 (no licence stated), at its source, the page of Alex Krizhevsky at the University of Toronto, which sends
    # the download on to cave.cs.toronto.edu: its binary version, bytes only (course/route/bronnen.yaml,
    # cifar-10-download-only; everything_ai.cifar unpacks it). The page gives the same file's MD5,
    # c32a1d4ab5d03f1284b67883e8d87530.
    ("cs.toronto.edu/kriz", "cifar-10-binary.tar.gz"): {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "revision": "v1",
        "sha256": "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
        "size": 170_052_171,
    },
    # The route database of OpenFlights (ODbL 1.0), at its source, the repository of OpenFlights on GitHub, at a fixed
    # commit: ml-network-data and the lessons after it (course/route/bronnen.yaml, openflights-routes;
    # everything_ai.flights reads it). A text file of 67,663 lines, read as CSV: nothing in it is ever run.
    ("github.com/jpatokal/openflights", "routes.dat"): {
        "url": "https://raw.githubusercontent.com/jpatokal/openflights/cd163fa17710f3135540651626ad719dd5322a05/data/routes.dat",
        "revision": "cd163fa17710f3135540651626ad719dd5322a05",
        "sha256": "bd373706238134f619c624c606dccc74c05c2582a977c489c81de501735f2390",
        "size": 2_377_148,
    },
    # The SMS Spam Collection v.1 (CC BY 4.0), at its source, the UCI Machine Learning Repository, dataset 228:
    # nlp-project-spam and the lessons after it (course/route/bronnen.yaml, sms-spam-collection; everything_ai.sms
    # reads it). A zip with two text files, of which the course reads one as text: nothing in it is ever run. UCI keeps
    # no versions of the file, so "v1" is the course's own name for the file with this SHA-256.
    ("archive.ics.uci.edu/dataset/228", "sms+spam+collection.zip"): {
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "revision": "v1",
        "sha256": "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
        "size": 203_415,
    },
    # The test set of SciTLDR-A (Apache-2.0, Cachola and others, 2020), from its repository at a fixed commit:
    # nlp-summarize and the lessons after it (course/route/bronnen.yaml, scitldr; everything_ai.scitldr reads it). A text
    # file of 618 lines, one JSON object each, read as text: nothing in it is ever run.
    ("github.com/allenai/scitldr", "test.jsonl"): {
        "url": "https://raw.githubusercontent.com/allenai/scitldr/5ccad9c00a60ad75c9e04abf7f27d0f53f983b20/SciTLDR-Data/SciTLDR-A/test.jsonl",
        "revision": "5ccad9c00a60ad75c9e04abf7f27d0f53f983b20",
        "sha256": "fb42dd6cd4f4a1928ae8a01a189456fbfe994a07e938bd49f68653933f6503c9",
        "size": 1_204_107,
    },
    # SciFact (Wadden and others, EMNLP 2020) in the copy of BEIR on the hub, at a fixed commit: 5,183 abstracts of biomedical
    # papers, 1,109 claims and which abstracts are relevant to each, for ir-bm25 (course/route/bronnen.yaml, scifact;
    # everything_ai.scifact reads them). Text files, read as text: nothing in them is ever run.
    ("datasets/mteb/scifact", "corpus.jsonl"): {
        "revision": "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
        "sha256": "f0d32db0d156b526d75921ed7a76f2cb912902631c87248c5c97c617bad0b60c",
        "size": 8_023_638,
    },
    ("datasets/mteb/scifact", "queries.jsonl"): {
        "revision": "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
        "sha256": "9db7df096f7414435d52bafcbacf814c30cea50eb565c1e8fa6d11440759bba8",
        "size": 129_085,
    },
    ("datasets/mteb/scifact", "qrels/train.tsv"): {
        "revision": "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
        "sha256": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
        "size": 14_502,
    },
    ("datasets/mteb/scifact", "qrels/test.tsv"): {
        "revision": "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
        "sha256": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
        "size": 5_389,
    },
    # BERT-Tiny, the smallest BERT of Google's miniatures (Apache-2.0, Turc and others, released 2020), with the head that fills in
    # masked words, and its WordPiece vocabulary: nlp-static-vs-contextual and later lessons on BERT
    # (course/route/bronnen.yaml, model-bert-tiny; everything_ai.bert writes the model and the tokenizer out)
    ("google/bert_uncased_L-2_H-128_A-2", "model.safetensors"): {
        "revision": "30b0a37ccaaa32f332884b96992754e246e48c5f",
        "sha256": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
        "size": 17_739_144,
    },
    ("google/bert_uncased_L-2_H-128_A-2", "vocab.txt"): {
        "revision": "30b0a37ccaaa32f332884b96992754e246e48c5f",
        "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        "size": 231_508,
    },
    # The vectors of fastText for Dutch, cc.nl.300 (CC BY-SA 3.0, Grave and others 2018), at their source, the downloads of
    # fastText: the first 20 MB of the gzipped file, the 30,355 most frequent words (course/route/bronnen.yaml,
    # fasttext-nl-30k; everything_ai.wordvectors reads them as text: nothing in them is ever run). The file has no
    # versions, so "v1" is the course's own name for the file whose first 20 MB have this SHA-256.
    ("dl.fbaipublicfiles.com/fasttext", "cc.nl.300.vec.gz"): {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz",
        "revision": "v1",
        "sha256": "8c59089d944e31e6b9f9496d67851ffca20f51b7ff69f2618c6e628a821f8995",
        "size": 20_000_000,
        "first": True,
    },
}

# NTREX-128 (CC BY-SA 4.0, Federmann, Kocmi and Xin 2022), from its repository at a fixed commit: 1,997 sentences of news
# from WMT19 in English, translated by people into 128 languages, one sentence per line (course/route/bronnen.yaml,
# ntrex-128; everything_ai.ntrex reads them). The English source and the Dutch whole; of the other languages only the
# first 100 lines, which is all a lesson uses. Text files, read as text: nothing in them is ever run.
NTREX = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"
for _code, _sha256, _size, _first in (   # the language's code in NTREX, and the SHA-256 and size of what is pinned
    ("eng", "389e8f5796c66db4f646dfad33e1ec622d74767af5ef112b42a1f2cd814df3cc", 251_739, False),
    ("afr", "2fc18c704f213c022a4b713c82d405f8ae8d7b7223e385e44e83ce989c786bdc", 14_182, True),
    ("amh", "9c80f82734115e724bc09da2ddbeebf0ea5e37453b41c9952a6720b46f0fd43d", 23_694, True),
    ("arb", "c08ca658d923e3d0c1a428e471d99060349947cc3d096ccae248495daea7310b", 22_297, True),
    ("ben", "9d48413cafcdde1b132b700f743e7797736aa57745970397dd23fb82ffa03e6c", 34_853, True),
    ("bod", "9111062b63327fe1c15b5e274f1de90d8fc1fc9bc42970d06007adee296e0e53", 43_519, True),
    ("cym", "15424fb6f6d2fddf64835a60cb253f6cf4173d59c609ad722f894c61db14d735", 14_533, True),
    ("deu", "c107af300b9809119f8b4ea647a7920493ab525dccee3ede64c2e573ed0a4de7", 16_153, True),
    ("ell", "b06f3da80f67fe2e22655b651ccdfc5c63172ed6e4c712f91c7b590349e424c5", 28_203, True),
    ("eus", "8a3907ade41c5d9c5571d43746621c20fe706e6e18a23363a8b3d92e00c2554a", 15_045, True),
    ("ewe", "f1a77b50f3e9b4c4786978644d1eb4d374e08561310b42ee2b169a43e2c73bd7", 13_522, True),
    ("fao", "bb303f978e210e89d14849ad2cf04088f48fce825eb1145ac7fff820fbe31f57", 14_468, True),
    ("fas", "8181a92c0feba8a31ef2f279c93c894a8c76044d4b58f5093dfc161616e77fab", 22_694, True),
    ("fin", "96afbd8ec885025cf2cc6f550b6396215c48e676bb096909db8cdc2341de0cf2", 14_558, True),
    ("fra", "f7a1b318ca2392919db0c55ac9862eb6d3a325523e86c31ee0793ce3b4c237c8", 16_280, True),
    ("gle", "ffff57a1acf0d824325860b936696dbe9622e75510ce7b3e993ee8ea65dea465", 16_481, True),
    ("hau", "af1b6a8a93c6437c10ab5e007cbbb4e3c31237347e7484a6b19ce3291c544b63", 15_306, True),
    ("heb", "db0c6a27fe1540f3f4cce3369af1746c21dd4326396690b059c8dc9bc147c309", 18_967, True),
    ("hin", "f88d23d882a4822d9acf59f1f077426337e172acfac77639cf7fa1578aa16cda", 35_569, True),
    ("hun", "835578f4a40f80126f3d358c8d2efa6d900a6204d8191240efa3f40cefd67597", 16_091, True),
    ("hye", "b4c24cd27af70136a38e35548a37250accb832ba1ce8102a813a0b59ea5c35fc", 26_550, True),
    ("ind", "ae480ae40cdcc09eae22faa0b86f7a217e1b197f740a2b450c7c8c40a9c5f822", 15_586, True),
    ("isl", "6ec2ad38c6c7b04f6f50a78863c5d44090876532ca96ac3fdfadae59f4540280", 15_307, True),
    ("jpn", "0412cf05906c0b7ccb6d4f864517c6d3c5e623a2eb9778d07565d937da9d25ca", 15_826, True),
    ("kat", "54f58f6d4df8c919ed638d4f3468251d18b81e163839a2303ea6249b8e761872", 35_960, True),
    ("khm", "fe90aec61b9f590ea451d567701161e8b4b2619b8a6604ae2e50a63d2955e4f6", 37_648, True),
    ("kor", "0b1ff8ccb3e9800676036157d13b53b8d77269ce4f87168e27c26dcbd083f8fe", 15_958, True),
    ("lao", "37e1be8aa68c569aa3e146ebb1d619f171c7994611aa5b455bf15638c4d5a762", 33_495, True),
    ("ltz", "2dfa0364108bf9bc7c8f86df6841d69d2d05385384bd3ca3c3b006021fea794f", 14_947, True),
    ("mri", "f7cebec0fb9bee1e5c647d77897f963e95e030c0e14fbf9a889217100b9482b6", 14_909, True),
    ("nld", "dce60b592d410932cb369301d0506f1b3d0ae4f7bedf0369a3cdf870f462b4a5", 297_134, False),
    ("pol", "21349681b8bc6302512152a50ffeae98b28046445374377c0f29f2d80e9a6100", 16_586, True),
    ("rus", "9d38b4000d2fc7a7349971f37f492f17aacae60c67af4da8926cad4479dda2b0", 26_130, True),
    ("smo", "1545d3b767f02882358a7e9f644a814c49cc74dc92a974e3c8b0fd7d8ba8b36f", 13_961, True),
    ("som", "3bf3e059ca5b00179ce5d22bcca8de3f1a88b34224ff423b09dd4211922f897b", 14_644, True),
    ("spa", "ebf51da7e645c67e66f9b5babe8b26d5b809403fe70df24293a9278351cd1694", 15_549, True),
    ("swa", "93142d01e176c7b5140f143bc86b6b47d1908c2f073e09654b50d74bbc2661fd", 15_176, True),
    ("tam", "b6169c97239967dd3c2f8d0255cd6e1c4e9ade0c8f433a88f856fecfecf4aefe", 39_878, True),
    ("tel", "2360215c17d30afa79a31b27aaaedd1580ebf22ac3fb8295dd4ca69afe1eff67", 36_707, True),
    ("tha", "57e85e4f22565432f5ff6299d28d08b57c6b40325363eceb75fb627b7140e9ac", 36_648, True),
    ("tir", "5eaf522ffc38299f704ed2c6e02a3da26346dea981706be4e21533892627fa88", 23_135, True),
    ("tur", "6a598f435e3d0d396a4b086f4c14e6a9a3b36524146f7409cd9ac42a731f4c56", 14_344, True),
    ("urd", "63d19d6f3e2e35f4a90fd3f6a62f497cc9d6042c3c0da0fec32775384a79dc7f", 23_197, True),
    ("vie", "935fb13ed0a55a61dc8117ba2b7fdccc861b5ba4079581d2a4c2ee7590e31ec2", 19_255, True),
    ("wol", "11a8af1b5b9dad22ff128e3a7a57e4903afa92a1b061bc97dedafc2d03b99c3d", 12_881, True),
    ("yor", "3f857b082aec9d0cfeba59a75c77df3b6f0bc661969c9f9603ad1bf8fd574a3e", 22_561, True),
    ("zho-CN", "4e6bb20c79757dbf06fafaf8d174b09a53b1dd5ed4fcc64332825f2762d42693", 12_960, True),
    ("zul", "d7f63f1f0ab923e6748de3e895a38d3352b7efb64d09006b735cd2137bd41ca4", 12_551, True),
):
    _name = "newstest2019-src.eng.txt" if _code == "eng" else f"newstest2019-ref.{_code}.txt"
    PINNED[("github.com/MicrosoftTranslator/NTREX", _name)] = {
        "url": f"https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{NTREX}/NTREX-128/{_name}",
        "revision": NTREX, "sha256": _sha256, "size": _size, **({"first": True} if _first else {}),
    }
del _code, _sha256, _size, _first, _name
CHUNK = 1 << 20   # bytes per read while downloading
AGENT = "everything-ai-course"   # how a download introduces itself: some sources refuse Python's own name


def address(repo: str, filename: str) -> str:
    """The fixed address of a file on the hub: the newest version on the branch main."""
    return f"{HUB}/{repo}/resolve/main/{filename}"


def page(repo: str) -> str:
    """The web page of a repository on the hub, with its model card."""
    return f"{HUB}/{repo}"


_UNREACHABLE: set[str] = set()   # repositories this process could not download from: it does not wait for them again


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(repo: str, filename: str) -> tuple[Path, str]:
    """A file of one of the course's repositories, and where it came from: "hub" when it was downloaded from the hub,
    now or earlier, and "course" when the course's own copy is used."""
    copy = REPOS[repo][filename]
    want = _sha256(copy)
    cached = paths.WORK / "cache" / "hub" / repo / filename
    if cached.is_file() and _sha256(cached) == want:
        return cached, "hub"
    if os.environ.get("EAI_OFFLINE") or repo in _UNREACHABLE:
        return copy, "course"
    try:
        with urllib.request.urlopen(address(repo, filename), timeout=TIMEOUT) as response:
            data = response.read(copy.stat().st_size + 1)   # never more than the course's file and one byte
    except Exception:   # no connection, no such repository yet, or the hub does not answer
        _UNREACHABLE.add(repo)
        return copy, "course"
    if hashlib.sha256(data).hexdigest() != want:
        return copy, "course"
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cached.parent, prefix=filename + ".", suffix=".part")   # one of its own per lesson
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)   # mkstemp makes a file only its owner may read
        os.replace(tmp, cached)   # in one step, also over an old file: another lesson never reads half a file
    except OSError:   # a work folder the course cannot write to: the file is still right, only not kept
        return copy, "hub"
    return cached, "hub"


class NotPinned(OSError):
    """The hub gave another file than the one the course pinned: another size or SHA-256."""


def pinned_address(repo: str, filename: str) -> str:
    """The address of a pinned file on the hub: at its fixed commit, so that it never changes. A file from elsewhere
    has its own address."""
    pin = PINNED[(repo, filename)]
    return pin.get("url") or f"{HUB}/{repo}/resolve/{pin['revision']}/{filename}"


def _pinned_path(repo: str, filename: str) -> Path:
    pin = PINNED[(repo, filename)]
    name = f"{filename}.first{pin['size']}" if pin.get("first") else filename   # only the beginning of the file
    return paths.WORK / "cache" / "hub" / repo / pin["revision"] / name


def pinned(repo: str, filename: str) -> Path | None:
    """The downloaded file, when it is there with its full size, else None. It looks only on disk: the SHA-256 was
    checked when the file was downloaded."""
    path = _pinned_path(repo, filename)
    return path if path.is_file() and path.stat().st_size == PINNED[(repo, filename)]["size"] else None


def download(repo: str, filename: str, progress=None) -> Path:
    """Download a pinned file once, into work/cache/hub/, and give its path. `progress(done, total)` hears about every
    megabyte. A file of another size or SHA-256 is thrown away: then it raises NotPinned, and without a connection, or
    when it breaks off, another OSError. EAI_OFFLINE makes it never try."""
    have = pinned(repo, filename)
    if have is not None:
        return have
    if os.environ.get("EAI_OFFLINE"):
        raise OSError("offline: EAI_OFFLINE is set")
    pin = PINNED[(repo, filename)]
    target = _pinned_path(repo, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
    try:
        digest, done = hashlib.sha256(), 0
        first = pin.get("first", False)
        headers = {"User-Agent": AGENT}
        if first:   # ask for those bytes only
            headers["Range"] = f"bytes=0-{pin['size'] - 1}"
        request = urllib.request.Request(pinned_address(repo, filename), headers=headers)
        # of a file pinned by its first bytes, exactly those: also when the hub sends the whole file anyway
        wanted = pin["size"] if first else pin["size"] + 1
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(request, timeout=60) as response:
            while done < wanted:   # never more than the pinned size and one byte
                chunk = response.read(min(CHUNK, wanted - done))
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, pin["size"])
        if done != pin["size"] or digest.hexdigest() != pin["sha256"]:
            raise NotPinned(f"the downloaded file is not the pinned one ({done} bytes)")
        os.chmod(tmp, 0o644)   # mkstemp makes a file only its owner may read; a download is for whoever uses the work folder
        os.replace(tmp, target)   # in one step: another lesson never reads half a file
        return target
    except BaseException as e:
        Path(tmp).unlink(missing_ok=True)
        if isinstance(e, http.client.HTTPException):   # a response that broke off in the middle
            raise ConnectionError(f"the connection broke off ({type(e).__name__})") from e
        raise
