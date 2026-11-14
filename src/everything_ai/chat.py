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
NO_THINKING = "<think>\n\n</think>\n\n"   # an empty thought after the generation prompt: non-thinking mode
TOOLS_HEAD = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TOOLS_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
              "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
              "<args-json-object>}\n</tool_call>")
# How an answer is sampled: Qwen3's advice for non-thinking mode (README of Qwen/Qwen3-0.6B). Greedy decoding makes a
# small model repeat itself, so the course samples, with a seed, and records what came out.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
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
    a chat that ends with a message of the user or of a tool."""
    out, messages = [], list(messages)
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
    out.append(f"{IM_START}assistant\n" + ("" if thinking else NO_THINKING))
    return "".join(out)


def parse(text: str) -> dict:
    """An answer of the model as a message: {"role": "assistant", "content": its text without thinking and tool calls,
    "tool_calls": every call that is valid JSON with a "name" and a dict of arguments, as {"name": ..., "arguments": {...}}, "bad_calls": the
    text of every other call, also one the answer broke off in}."""
    text = THOUGHT.sub("", text)
    calls, bad = [], []
    for raw in CALL.findall(text):
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
    content = CALL.sub("", text)
    if "<tool_call>" in content:   # a call that never closed
        content, rest = content.split("<tool_call>", 1)
        bad.append(rest.strip())
    return {"role": "assistant", "content": content.strip(), "tool_calls": calls, "bad_calls": bad}


class Chat:
    """Qwen3-0.6B, ready to answer: the chat model in float32 on the CPU, or the model and tokenizer given."""

    def __init__(self, model=None, enc=None):
        from everything_ai import qwen3
        self.model = model if model is not None else qwen3.load(repo=qwen3.CHAT)
        self.enc = enc if enc is not None else qwen3.tokenizer()
        self.stops = {self.enc.encode_single_token(IM_END), self.enc.encode_single_token(END)}

    def tokens(self, text: str) -> list:
        from everything_ai import qwen3
        return qwen3.encode(self.enc, text)

    def answer(self, messages: list, tools: list | None = None, max_tokens: int = 256, seed: int = 0,
               temperature: float = SAMPLING["temperature"], top_p: float = SAMPLING["top_p"], top_k: int = SAMPLING["top_k"],
               thinking: bool = False, heard=None) -> dict:
        """The model's answer to `messages`: {"text": what it wrote, "message": parse(text), "prompt_tokens": ...,
        "tokens": tokens it wrote, "seconds": ..., "finish": "stop" when it ended its answer itself, "length" when it
        reached max_tokens}. Sampled with `seed` from the top_k most likely tokens that together reach top_p, at
        `temperature`; greedy at temperature 0. The whole prompt goes into an empty KV cache at once, then one token at
        a time. `heard(text)` hears the answer so far after every token."""
        import torch
        ids = self.tokens(prompt(messages, tools, thinking))
        device = next(self.model.parameters()).device
        g = torch.Generator(device=device).manual_seed(seed)
        out, finish, start = [], "length", time.perf_counter()
        with torch.no_grad():
            caches = [{} for _ in self.model.blocks]
            logits = self.model(torch.tensor([ids], device=device), caches, last=True)[0, -1].float()
            for _ in range(max_tokens):
                if temperature == 0:
                    nxt = int(logits.argmax())
                else:
                    top, which = torch.topk(logits / temperature, min(top_k, logits.numel()))
                    p = torch.softmax(top, -1)
                    p = torch.where(torch.cumsum(p, -1) - p < top_p, p, torch.zeros_like(p))   # the smallest set that reaches top_p
                    nxt = int(which[torch.multinomial(p / p.sum(), 1, generator=g)])
                if nxt in self.stops:
                    finish = "stop"
                    break
                out.append(nxt)
                if heard is not None:
                    heard(self.enc.decode(out))
                logits = self.model(torch.tensor([[nxt]], device=device), caches)[0, -1].float()
        text = self.enc.decode(out)
        return {"text": text, "message": parse(text), "prompt_tokens": len(ids), "tokens": len(out),
                "seconds": round(time.perf_counter() - start, 2), "finish": finish}


# --- running it in a lesson: once loaded, the answer so far shown while it writes ------------------------------------

_LOADED = {}


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
    except MemoryError:
        return "memory"
    except Exception:
        return "load"
    shown = [0.0]

    def heard(text):
        if time.perf_counter() - shown[0] > every:
            shown[0] = time.perf_counter()
            show(text)
    try:
        return model.answer(messages, heard=heard, **answer)
    except MemoryError:
        return "memory"


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
