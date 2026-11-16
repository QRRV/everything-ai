# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Chatting with Qwen3-0.6B, the course's local chat model (everything_ai.qwen3, repo CHAT): the chat template that
turns messages and tools into one text (`prompt`), reading the text and the tool calls of an answer (`parse`), and
writing an answer token by token (`Chat.answer`). The template is Qwen3's own, from the tokenizer_config.json of
Qwen/Qwen3-0.6B at the commit the course pins, written out so that a lesson can show every part of it; by default in
non-thinking mode, which writes no reasoning before the answer and so fits a CPU. Importing this module does not
import PyTorch: `prompt` and `parse` work without a model, as in the checks of the lessons."""
import json
import re
import time

IM_START, IM_END, END = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
# The chat model's files, as everything_ai.qwen3.CHAT_FILES has them: the weights of Qwen3-0.6B, the tokenizer of the Base
FILES = [("Qwen/Qwen3-0.6B", "model.safetensors"), ("Qwen/Qwen3-0.6B-Base", "vocab.json"),
         ("Qwen/Qwen3-0.6B-Base", "tokenizer_config.json")]
NO_THINKING = "<think>\n\n</think>\n\n"   # an empty thought after the generation prompt: non-thinking mode
TOOLS_HEAD = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TOOLS_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
              "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
              "<args-json-object>}\n</tool_call>")
# How an answer is sampled: Qwen3's advice for non-thinking mode, and for thinking mode (README of Qwen/Qwen3-0.6B).
# Greedy decoding makes a small model repeat itself, so the course samples, with a seed, and records what came out.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THOUGHT = re.compile(r"<think>.*?</think>\s*", re.S)


def tool(name: str, description: str, **arguments: str) -> dict:
    """A tool in the form the chat template and OpenAI's API take: a function with a name, a description and its
    arguments, each a string described in a few words. tool("get_weather", "The weather now in a city.", city="the
    name of the city") is enough for the lessons; a tool with numbers or lists writes out its JSON Schema itself."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in arguments.items()},
                       "required": list(arguments)}}}


def _call_text(call: dict) -> str:
    call = call.get("function", call)
    args = call["arguments"] if isinstance(call["arguments"], str) else json.dumps(call["arguments"], ensure_ascii=False)
    return '<tool_call>\n{"name": "' + call["name"] + '", "arguments": ' + args + '}\n</tool_call>'


def prompt(messages: list, tools: list | None = None, thinking: bool = False) -> str:
    """The text the model reads for `messages`, each {"role": "system" | "user" | "assistant" | "tool", "content": ...},
    an assistant message perhaps with "tool_calls"; with `tools`, their descriptions in the system message; and at
    the end the start of the assistant's answer, which the model goes on from. As Qwen3's chat template writes it for
    a chat that ends with a message of the user or of a tool. A chat that ends with a message of the assistant without
    tool calls is a prefill: the answer starts with that text, and the model goes on from its last word, without
    thinking first, as Qwen3's template writes it then (its thought is empty)."""
    out, messages = [], list(messages)
    prefill = None
    if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
        prefill, messages = messages[-1].get("content") or "", messages[:-1]
    if tools:
        system = messages[0]["content"] + "\n\n" if messages and messages[0]["role"] == "system" else ""
        described = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
        out.append(f"{IM_START}system\n{system}{TOOLS_HEAD}{described}{TOOLS_TAIL}{IM_END}\n")
        if messages and messages[0]["role"] == "system":
            messages = messages[1:]
    for i, m in enumerate(messages):
        role, content = m["role"], m.get("content") or ""
        if role == "tool":   # tool results go back as the user's, between <tool_response> tags; a row of them in one turn
            first = i == 0 or messages[i - 1]["role"] != "tool"
            last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
            out.append((f"{IM_START}user" if first else "") + f"\n<tool_response>\n{content}\n</tool_response>"
                       + (f"{IM_END}\n" if last else ""))
        elif role == "assistant":
            calls = [("\n" if (j == 0 and content) or j > 0 else "") + _call_text(c) for j, c in enumerate(m.get("tool_calls") or [])]
            out.append(f"{IM_START}assistant\n{content}{''.join(calls)}{IM_END}\n")
        else:
            out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    out.append(f"{IM_START}assistant\n" + ("" if thinking and prefill is None else NO_THINKING) + (prefill or ""))
    return "".join(out)


def parse(text: str) -> dict:
    """An answer of the model as a message: {"role": "assistant", "content": its text without thinking and tool calls,
    "tool_calls": every call that is valid JSON with a "name" and a dict of arguments, as {"name": ..., "arguments": {...}}, "bad_calls": the
    text of every other call, also one the answer broke off in}. A last call without its closing tag counts when its
    JSON is whole, as vLLM's parser of this form of tool call reads it: the model ended its answer after it."""
    text = THOUGHT.sub("", text)
    calls, bad = [], []

    def read(raw: str) -> None:
        try:
            c = json.loads(raw)
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                raise ValueError
            args = c.get("arguments", {})
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise ValueError
            calls.append({"name": c["name"], "arguments": args})
        except (ValueError, TypeError):
            bad.append(raw)
    for raw in CALL.findall(text):
        read(raw)
    content = CALL.sub("", text)
    if "<tool_call>" in content:   # a call that never closed
        content, rest = content.split("<tool_call>", 1)
        read(rest.strip())
    return {"role": "assistant", "content": content.strip(), "tool_calls": calls, "bad_calls": bad}


def _nucleus(logits, temperature: float, top_k: int, top_p: float):
    """The tokens to draw the next one from: the top_k most likely at `temperature`, of which the smallest set that
    together reaches top_p, always at least the most likely one. Their probabilities (zero for the rest) and ids."""
    import torch
    top, which = torch.topk(logits / temperature, min(top_k, logits.numel()))
    p = torch.softmax(top, -1)
    keep = torch.cumsum(p, -1) - p < top_p
    keep[0] = True
    return torch.where(keep, p, torch.zeros_like(p)), which


# --- JSON mode: the answer can only become one JSON object ------------------------------------------------------------

_NUM = {   # the states of a JSON number while it is written, and where each next character takes it
    "start": {"-": "minus", "0": "zero", **dict.fromkeys("123456789", "int")},
    "minus": {"0": "zero", **dict.fromkeys("123456789", "int")},
    "zero": {".": "dot", "e": "exp", "E": "exp"},
    "int": {**dict.fromkeys("0123456789", "int"), ".": "dot", "e": "exp", "E": "exp"},
    "dot": dict.fromkeys("0123456789", "frac"),
    "frac": {**dict.fromkeys("0123456789", "frac"), "e": "exp", "E": "exp"},
    "exp": {"+": "sign", "-": "sign", **dict.fromkeys("0123456789", "power")},
    "sign": dict.fromkeys("0123456789", "power"),
    "power": dict.fromkeys("0123456789", "power"),
}
_NUM_WHOLE = {"zero", "int", "frac", "power"}   # a number may end in these states
_LITERALS = ("true", "false", "null")
_HEX = set("0123456789abcdefABCDEF")


def json_object_state(text: str) -> str | None:
    """How far `text` is on its way to being one JSON object, as JSON mode checks it after every token: "done" when it
    is one whole object, "open" when more text can still make it one, None when no text can. White space may come
    before and after the object, and between its parts, as JSON allows."""
    stack = []        # the objects and lists that are open: "{" or "["
    want = "start"    # what may come next: start, key, key_or_end, colon, value, value_or_end, after (a value), done
    string = None     # inside a string: whether it is a key, whether a backslash came, how many hex digits of \u are left
    number = None     # inside a number: its state in _NUM
    literal = None    # inside true, false or null: what is written of it so far
    for ch in text:
        if string is not None:
            if string["hex"]:
                if ch not in _HEX:
                    return None
                string["hex"] -= 1
            elif string["escape"]:
                if ch == "u":
                    string["hex"] = 4
                elif ch not in '"\\/bfnrt':
                    return None
                string["escape"] = False
            elif ch == "\\":
                string["escape"] = True
            elif ch == '"':
                want, string = ("colon" if string["key"] else "after"), None
            elif ord(ch) < 0x20:   # a line break or tab in a string must be written as \n or \t
                return None
            continue
        if number is not None:
            if ch in _NUM[number]:
                number = _NUM[number][ch]
                continue
            if number not in _NUM_WHOLE:
                return None
            number, want = None, "after"   # the number is over: this character comes after it
        if literal is not None:
            literal += ch
            if not any(w.startswith(literal) for w in _LITERALS):
                return None
            if literal in _LITERALS:
                literal, want = None, "after"
            continue
        if ch in " \t\n\r":
            continue
        if want == "start" and ch == "{":
            stack.append("{")
            want = "key_or_end"
        elif want in ("key", "key_or_end") and ch == '"':
            string = {"key": True, "escape": False, "hex": 0}
        elif want == "colon" and ch == ":":
            want = "value"
        elif want in ("value", "value_or_end") and ch in '{["' or want in ("value", "value_or_end") and ch in _NUM["start"]:
            if ch == "{":
                stack.append("{")
                want = "key_or_end"
            elif ch == "[":
                stack.append("[")
                want = "value_or_end"
            elif ch == '"':
                string = {"key": False, "escape": False, "hex": 0}
            else:
                number = _NUM["start"][ch]
        elif want in ("value", "value_or_end") and ch in "tfn":
            literal = ch
        elif want == "after" and ch == ",":
            want = "key" if stack[-1] == "{" else "value"
        elif (want == "after" and ch == {"{": "}", "[": "]"}[stack[-1]] or want == "key_or_end" and ch == "}"
              or want == "value_or_end" and ch == "]"):
            stack.pop()
            want = "after" if stack else "done"
        else:
            return None
    return "done" if want == "done" else "open"


def shared_start(a: list, b: list) -> int:
    """How many tokens two lists of tokens have in common at the start: up to the first place where they differ."""
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


class Chat:
    """Qwen3-0.6B, ready to answer: the chat model in float32 on the CPU, or the model and tokenizer given."""

    def __init__(self, model=None, enc=None):
        from everything_ai import qwen3
        self.model = model if model is not None else qwen3.load(repo=qwen3.CHAT)
        self.enc = enc if enc is not None else qwen3.tokenizer()
        self.stops = {self.enc.encode_single_token(IM_END), self.enc.encode_single_token(END)}
        self.kept = None   # with cache=True: the tokens of the last answer, its prompt and what it wrote, and their KV cache

    def tokens(self, text: str) -> list:
        from everything_ai import qwen3
        return qwen3.encode(self.enc, text)

    def answer(self, messages: list, tools: list | None = None, max_tokens: int = 256, seed: int = 0,
               temperature: float = SAMPLING["temperature"], top_p: float = SAMPLING["top_p"], top_k: int = SAMPLING["top_k"],
               thinking: bool = False, heard=None, stop: list | None = None, json_object: bool = False,
               cache: bool = False, keep: bool | None = None, logprobs: int | None = None) -> dict:
        """The model's answer to `messages`: {"text": what it wrote, "message": parse(text), "prompt_tokens": ...,
        "tokens": tokens it wrote, "seconds": ..., "finish": "stop" when it ended its answer itself, "length" when it
        reached max_tokens}. Sampled with `seed` from the top_k most likely tokens that together reach top_p, at
        `temperature`; greedy at temperature 0. The whole prompt goes into an empty KV cache at once, then one token at
        a time. `heard(text)` hears the answer so far after every token. With `stop`, a list of texts, the model stops
        writing as soon as its answer (after the thought, in thinking mode) holds one of them: finish "stop". With
        `json_object` (JSON mode, without thinking), every token must keep the answer on its way to one JSON object, a
        prefill included: a sampled token that does not is put aside and another drawn, and the answer ends as soon
        as the object is whole. Prompt caching: with `keep` the KV cache of this answer is kept, and with `cache` the
        tokens this prompt has in common at the start with the kept one are not computed again ("cached_tokens" says
        how many there were); `keep` is `cache` unless given. With `logprobs`, a whole number from 0 to 20, the answer
        also has "logprobs": for every token written, its log probability, and that of the `logprobs` most likely
        tokens at that place, as OpenAI's API gives them (`logprob_entry`), from the model's own probabilities, before
        temperature, top_k and top_p."""
        import torch
        last = messages[-1] if messages else {}
        prefilled = last.get("role") == "assistant" and not last.get("tool_calls")
        thinking = thinking and not prefilled   # after a prefill the model does not think: prompt() leaves the thought empty
        begun = (last.get("content") or "") if prefilled else ""
        ids = self.tokens(prompt(messages, tools, thinking))
        device = next(self.model.parameters()).device
        g = torch.Generator(device=device).manual_seed(seed)
        out, finish, start = [], "length", time.perf_counter()
        chances = [] if logprobs is not None else None
        with torch.no_grad():
            caches, cached = [{} for _ in self.model.blocks], 0
            if cache and self.kept is not None:
                cached = shared_start(self.kept[0], ids)
                cached = min(cached, len(ids) - 1)   # at least the last token goes in, for the logits of the next one
                if cached > 0:   # the keys and values of those tokens, from the kept cache
                    caches = [{"k": c["k"][:, :, :cached], "v": c["v"][:, :, :cached]} for c in self.kept[1]]
            logits = self.model(torch.tensor([ids[cached:]], device=device), caches, last=True)[0, -1].float()
            for _ in range(max_tokens):
                if json_object:
                    nxt = self._json_token(logits, out, begun, temperature, top_k, top_p, g)
                    if nxt is None:   # no token keeps it JSON: it ends here
                        finish = "stop"
                        break
                elif temperature == 0:
                    nxt = int(logits.argmax())
                else:
                    p, which = _nucleus(logits, temperature, top_k, top_p)
                    nxt = int(which[torch.multinomial(p / p.sum(), 1, generator=g)])
                if nxt in self.stops:
                    finish = "stop"
                    break
                out.append(nxt)
                if chances is not None:
                    chances.append(self.logprob_entry(logits, nxt, logprobs))
                if heard is not None or stop:
                    so_far = self.enc.decode(out)
                    if heard is not None:
                        heard(so_far)
                    if stop:
                        answer = (so_far.split("</think>", 1)[1] if "</think>" in so_far else "") if thinking else so_far
                        answer = answer.split("<tool_call>", 1)[0]   # a stop sequence in the markup of a tool call does not count
                        if any(s and s in answer for s in stop):
                            finish = "stop"
                            break
                if json_object and json_object_state(begun + self.enc.decode(out)) == "done":
                    finish = "stop"
                    break
                logits = self.model(torch.tensor([[nxt]], device=device), caches)[0, -1].float()
        if cache if keep is None else keep:   # the prompt, and the tokens written that went back in: all but one that ended it
            self.kept = ((ids + out)[:caches[0]["k"].shape[2]], caches)
        text = self.enc.decode(out)
        a = {"text": text, "message": parse(text), "prompt_tokens": len(ids), "tokens": len(out),
             "seconds": round(time.perf_counter() - start, 2), "finish": finish, "cached_tokens": cached}
        if chances is not None:
            a["logprobs"] = chances
        return a

    def logprob_entry(self, logits, token: int, top: int) -> dict:
        """One token as OpenAI's API gives it under logprobs.content: {"token": its text, "logprob": the natural log of
        its probability among all next tokens, "bytes": the bytes of its text, "top_logprobs": the `top` most likely
        tokens at that place, most likely first, each with its token, logprob and bytes}. A token can be part of a
        character: its text then has the replacement character, and its bytes say what it is."""
        import torch
        logp = torch.log_softmax(logits.float(), dim=-1)

        def one(t: int) -> dict:
            b = self.enc.decode_single_token_bytes(t)
            return {"token": b.decode("utf-8", "replace"), "logprob": round(float(logp[t]), 6), "bytes": list(b)}
        n = getattr(self.enc, "n_vocab", None) or logp.shape[-1]   # rows past the tokenizer's last token are padding
        best = torch.topk(logp[:n], min(top, n)).indices.tolist() if top else []
        return {**one(token), "top_logprobs": [one(t) for t in best]}

    def _json_token(self, logits, out: list, begun: str, temperature: float, top_k: int, top_p: float, g,
                    furthest: int = 5000) -> int | None:
        """The next token in JSON mode: drawn as without it, but a token after which the answer (after `begun`, a
        prefill) can no longer become one JSON object is put aside, and another is drawn from the rest; the end of
        the answer only once the object is whole. When none of the tokens it may draw from will do, the most likely
        one of the `furthest` most likely that does; None when there is none."""
        import torch

        def fits(token: int) -> bool:
            if token in self.stops:
                return json_object_state(begun + self.enc.decode(out)) == "done"
            return json_object_state(begun + self.enc.decode(out + [token])) is not None
        if temperature != 0:
            p, which = _nucleus(logits, temperature, top_k, top_p)
            while float(p.sum()) > 0:
                i = int(torch.multinomial(p / p.sum(), 1, generator=g))
                if fits(int(which[i])):
                    return int(which[i])
                p[i] = 0.0
        for token in torch.argsort(logits, descending=True)[:furthest].tolist():
            if fits(token):
                return token
        return None


# --- running it in a lesson: once loaded, the answer so far shown while it writes ------------------------------------

_LOADED = {}


def memory_full(e: BaseException) -> bool:
    """Whether an error says the memory ran out: Python's MemoryError, or PyTorch's, which is a RuntimeError
    (torch.OutOfMemoryError on a GPU, "can't allocate memory" or "not enough memory" on a CPU)."""
    if isinstance(e, MemoryError):
        return True
    words = str(e).lower()
    return isinstance(e, RuntimeError) and (type(e).__name__ == "OutOfMemoryError" or any(
        w in words for w in ("out of memory", "can't allocate memory", "not enough memory")))


def loaded() -> Chat:
    """The chat model, loaded the first time a lesson asks for it, and then kept for the rest of the session."""
    if "chat" not in _LOADED:
        _LOADED["chat"] = Chat()
    return _LOADED["chat"]


def answer_live(messages: list, show, every: float = 0.2, **answer):
    """`loaded().answer(messages, **answer)`, with what the model wrote so far handed to `show(text)` at most once
    every `every` seconds, and `show(None)` first, while the model loads. The answer, or what went wrong: "memory"
    when the memory is full, "load" when loading the model failed otherwise."""
    show(None)
    try:
        model = loaded()
    except Exception as e:   # noqa: BLE001
        return "memory" if memory_full(e) else "load"
    shown = [0.0]

    def heard(text):
        if time.perf_counter() - shown[0] > every:
            shown[0] = time.perf_counter()
            show(text)
    try:
        return model.answer(messages, heard=heard, **answer)
    except Exception as e:
        if memory_full(e):
            return "memory"
        raise


def answer_picture(text: str, note: str, title: str) -> str:
    """An answer of the chat model in a lesson's demo, framed as the learner's own (everything_ai.yours.figure): the
    text in the colour of what a model writes, and a note under it."""
    import html

    from everything_ai import yours
    body = (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;'
            f'color:var(--ea-pred-text)">{html.escape(text)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(note)}</div>')
    return yours.figure(title, body, alt=title)


def demo(messages: list, show, words: dict, lang: str, max_tokens: int = 256, seed: int = 0, note=None) -> str:
    """A lesson's demo: `messages` to the chat model, sampled with `seed`, with the answer so far shown by `show(html)`
    at most five times a second. `words` has "title", "loading", "writing", "done" (with {tokens} and {seconds}),
    "load" and "memory", in the lesson's language; `note(text)`, when given, adds a sentence of the lesson's own about
    the answer after "done". The picture of the answer, or the message of what went wrong."""
    import html
    a = answer_live(messages, lambda text: show(answer_picture(text or "", words["loading"] if text is None else words["writing"], words["title"])),
                    max_tokens=max_tokens, seed=seed)
    if isinstance(a, str):
        return f'<p><em>{html.escape(words[a])}</em></p>'
    seconds = f"{a['seconds']:.1f}".replace(".", "," if lang == "nl" else ".")
    done = words["done"].format(tokens=a["tokens"], seconds=seconds)
    text = a["message"]["content"]
    return answer_picture(text, done + (" " + note(text) if note else ""), words["title"])


# --- the download, for the lessons that run the chat model ------------------------------------------------------------

def downloaded() -> bool:
    from everything_ai import qwen3
    return qwen3.chat_downloaded()


def size_mb() -> int:
    """The size of the chat model and its tokenizer together, in MB, as the lessons' button says it."""
    from everything_ai import hub, qwen3
    return round(sum(hub.PINNED[pair]["size"] for pair in qwen3.CHAT_FILES) / 1e6)


def download_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until the chat model is downloaded; a click downloads it, with
    the progress in the cell. `texts` has "ask", "busy" ({done} and {total} in MB), "no_net", "not_pinned", "no_room"
    and "failed" ({why}), in the lesson's language, as llm-reasoning-idea has for the Base."""
    import errno
    import html
    import urllib.error

    import marimo as mo

    from everything_ai import hub, qwen3
    from everything_ai.lesson import download_card
    failed = None
    if not downloaded() and button.value:
        total, shown = size_mb(), [-1]

        def progress(done, size):
            mb = int(done / 1e6)
            if mb // 20 != shown[0] // 20:   # every 20 MB
                shown[0] = mb
                mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
        try:
            qwen3.download_chat(progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not downloaded():
        mo.stop(True, download_card(texts["ask"], button, failed))
