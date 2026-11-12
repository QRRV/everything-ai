# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Your network: the network a learner builds, drawn from one pass of a scene through it.

The course shows what the learner builds, and redraws it when their code changes (besluit 1.16). For a network,
this is that picture: a box for every layer, as tall as its feature maps are large and as wide as it has channels,
with its channels above it; an arc where a layer gets the output of an earlier layer than the one before it, as in a
skip connection; and, when the network does not fit together, the layer where Python stops and what it said. One
layer can be chosen: under the network are its name, its shapes, its parameters and the first feature maps it makes
of the scene.

    tr = netview.trace(model, x)            # one pass of x (a batch; the pictures use its first item)
    netview.picture(tr, lang, chosen=3)     # HTML text; chosen counts the boxes from 0, None for none
    picture=netview.network("UNet", scene)  # under the check of an exercise: the learner's class, redrawn as they type

Nothing here trains, and the model is left as it was: the pass runs in eval mode, as a network is used, and every
module gets its own mode back; no weights or running statistics change. Which box feeds which is followed through
the tensors themselves, also through a torch.cat or a +, and over every tensor a layer gets, so a skip connection is
found wherever the learner writes it: stacked in forward, or handed to a block of their own as a second input. An
activation the learner calls as a function (torch.relu(self.conv(x))) is drawn with the box it follows, as an nn.ReLU
is, and its feature maps are those after it. Every colour is a token (var(--ea-*)); a feature map is a mask over a square in ink, so it follows the light and
the dark theme without a picture per theme. The words are in the picture's HTML, so they wrap on a phone, and the
row of boxes scrolls sideways instead of shrinking its numbers."""
from __future__ import annotations

import base64
import contextlib
import html
import io
import math
import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from everything_ai import course, yours
from everything_ai.check import Timeout

# Layers that keep the shape and are drawn with the box before them, as words after its name: activations,
# normalization, dropout, flatten. A box per ReLU would make a network twice as long without saying more.
_WITH_BEFORE = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.CELU, nn.GELU, nn.SiLU, nn.Mish,
                nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.Softmax, nn.LogSoftmax,
                nn.Dropout, nn.Dropout2d, nn.AlphaDropout, nn.Identity, nn.BatchNorm1d, nn.BatchNorm2d,
                nn.InstanceNorm2d, nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm, nn.Flatten)
_OPENED = (nn.Sequential, nn.ModuleList, nn.ModuleDict)   # opened up only as themselves: a block that subclasses
                                                        # them is one box
# layers that make the maps smaller or larger (pooling, upsampling, an up-convolution): a dotted edge
_RESIZE = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.ConvTranspose2d, nn.Upsample,
           nn.UpsamplingNearest2d, nn.UpsamplingBilinear2d, nn.PixelShuffle)
# activations called as a function, by the name torch gives them, and the layer they are drawn as
_ACTIVATIONS = {"relu": "ReLU", "relu_": "ReLU", "relu6": "ReLU6", "leaky_relu": "LeakyReLU",
                "leaky_relu_": "LeakyReLU", "elu": "ELU", "elu_": "ELU", "selu": "SELU", "celu": "CELU", "gelu": "GELU",
                "silu": "SiLU", "mish": "Mish", "hardswish": "Hardswish", "hardsigmoid": "Hardsigmoid",
                "sigmoid": "Sigmoid", "sigmoid_": "Sigmoid", "tanh": "Tanh", "tanh_": "Tanh", "softplus": "Softplus"}
_SHOWN_MAPS = 8          # feature maps shown of the chosen layer
_SHOWN_VALUES = 64       # values shown of a chosen layer that gives a vector
_LEVEL = 34              # pixels from one size of map to the next, half as large


@dataclass
class Layer:
    name: str                       # its name in the model: "conv1", "down.2", "features.3"
    module: nn.Module
    shape_in: tuple | None = None   # without the batch: (channels, height, width), or (features,)
    shape_out: tuple | None = None  # of its output, or of the first tensor of it when it gives several
    out: torch.Tensor | None = None  # that output for the first item of the batch
    params: int = 0                 # of its module (a module used twice shows them twice; the title counts once)
    with_it: list[str] = field(default_factory=list)   # the layers drawn with it: ["BatchNorm2d", "ReLU"]
    fed: tuple = ()                 # the boxes whose output reaches its input, -1 for the scene; () when not known
    joined: str | None = None       # how two or more of them were joined before it: "cat" (stacked) or "add" (added);
                                    # None also when they came in as inputs of their own
    done: bool = False              # False for the layer the pass stopped in


@dataclass
class Trace:
    layers: list[Layer]
    shape_in: tuple                 # the scene, without the batch
    params: int                     # of the whole model, every parameter once
    error: str | None = None        # what Python said when the pass stopped
    stopped_in: str | None = None   # the layer it stopped in; None when it stopped between two layers


def _units(model: nn.Module, prefix: str = "") -> list[tuple[str, nn.Module]]:
    """The layers to draw: the children of the model, a Sequential or ModuleList opened up; a block of the learner's
    own (a DoubleConv, a residual block) stays one box."""
    if type(model) in _OPENED or (not prefix and list(model.children())):
        out = []
        for name, child in model.named_children():
            full = f"{prefix}.{name}" if prefix else name
            out += _units(child, full) if type(child) in _OPENED else [(full, child)]
        return out
    return [(prefix or type(model).__name__, model)]


def _tensors(obj, depth: int = 0):
    """The tensors in what a layer gets or gives: a tensor, or a tuple, list or dict of them."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif depth < 3 and isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o, depth + 1)
    elif depth < 3 and isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o, depth + 1)


def _shape(t) -> tuple | None:
    return tuple(t.shape[1:]) if isinstance(t, torch.Tensor) and t.dim() >= 2 else None


def _first_map(t: torch.Tensor) -> torch.Tensor:
    """The output for the first item of the batch, as a copy: a later in-place operation does not change it."""
    return t[0].detach().float().cpu().clone()


class _Flow(TorchFunctionMode):
    """Where every tensor of the pass comes from. A layer's output comes from that layer; what any other operation
    makes comes from all the layers its tensors came from, so a layer knows which layers fed it, also through a
    torch.cat or a +. Tensors are known by id, with a weak reference to tell a new tensor at a reused id apart.

    An activation called as a function on the output of the layer that just finished, outside every layer, belongs
    to that layer: `post` keeps its name and what it gave, so the box shows the maps after it."""

    def __init__(self):
        super().__init__()
        self.of: dict[int, tuple] = {}   # id(tensor) -> (weak reference, frozenset of calls, joined)
        self.open_: list[str] = []       # the layers running now, the innermost last
        self.tip: tuple | None = None    # (weak reference, call) of the output of the layer that finished last
        self.post: dict[int, tuple] = {}  # call -> (names of the activations after it, the output after them)

    def get(self, t) -> tuple[frozenset, str | None]:
        hit = self.of.get(id(t))
        return (hit[1], hit[2]) if hit and hit[0]() is t else (frozenset(), None)

    def set(self, t, sources: frozenset, joined: str | None) -> None:
        try:
            self.of[id(t)] = (weakref.ref(t), sources, joined)
        except TypeError:   # a tensor that takes no weak reference: it is not followed
            pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        seen = [self.get(t) for t in _tensors((args, kwargs))]
        groups = {s for s, _ in seen if s}
        if groups:
            if len(groups) > 1:
                joined = "cat" if any(w in name for w in ("cat", "concat", "stack")) else "add"
            else:
                joined = next(j for s, j in seen if s)
            sources = frozenset().union(*groups)
            for t in _tensors(out):
                self.set(t, sources, joined)
        tip = self.tip
        if (name in _ACTIVATIONS and not self.open_ and tip and args and tip[0]() is args[0]
                and isinstance(out, torch.Tensor) and out.dim() >= 2):
            names, _ = self.post.get(tip[1], ([], None))
            self.post[tip[1]] = (names + [_ACTIVATIONS[name]], _first_map(out))
            self.tip = (weakref.ref(out), tip[1])
        return out


def _leave(flow: _Flow) -> None:
    """Take the flow off torch's stack of modes when a time limit stopped the pass on its way in or out of the
    `with`: else every later torch operation of the session would still go through it."""
    try:
        stack = torch.overrides._get_current_function_mode_stack()
        if stack and stack[-1] is flow:
            torch._C._pop_torch_function_stack()
    except Exception:   # a later torch without these: the with block takes it off as before
        pass


def trace(model: nn.Module, x: torch.Tensor) -> Trace:
    """One pass of x through the model, with a hook on every layer. Every call is a box, so a pooling layer used
    three times is three boxes. When the pass stops, the trace holds the layers until there and what Python said."""
    calls: list[Layer] = []
    hooks = []
    flow = _Flow()
    open_ = flow.open_
    for name, module in _units(model):
        def before(m, args, kwargs, name=name):
            open_.append(name)
            # every tensor it gets: a block of the learner's own may take the kept map as a second input
            seen = [flow.get(t) for t in _tensors((args, kwargs))]
            groups = {s for s, _ in seen if s}
            sources = frozenset().union(*groups)
            joined = next(j for s, j in seen if s) if len(groups) == 1 else None
            first = next(_tensors((args, kwargs)), None)
            calls.append(Layer(name, m, shape_in=_shape(first), params=sum(p.numel() for p in m.parameters()),
                               fed=tuple(sorted(sources)), joined=joined))   # calls for now; boxes after _fold

        def after(m, args, out, name=name):
            if open_ and open_[-1] == name:
                open_.pop()
            i = next(i for i in range(len(calls) - 1, -1, -1) if calls[i].name == name and not calls[i].done)
            layer = calls[i]
            layer.done = True
            first = next(_tensors(out), None)
            layer.shape_out = _shape(first)
            layer.out = _first_map(first) if layer.shape_out is not None else None
            for t in _tensors(out):
                flow.set(t, frozenset({i}), None)
            if not open_:
                flow.tip = (weakref.ref(first), i) if first is not None else None
        hooks += [module.register_forward_pre_hook(before, with_kwargs=True), module.register_forward_hook(after)]
    modes = [(m, m.training) for m in model.modules()]
    error = stopped_in = None
    try:
        model.eval()
        flow.set(x, frozenset({-1}), None)
        with torch.no_grad(), flow:
            model(x)
    except Timeout:   # the check's time limit is not the network's error
        raise
    except Exception as e:   # the learner's network may not fit together: that is what the picture then shows
        error = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
        stopped_in = open_[-1] if open_ else None
    finally:
        _leave(flow)
        for h in hooks:
            h.remove()
        for m, mode in modes:
            m.training = mode
    for i, (names, after_them) in flow.post.items():   # activations called as a function: drawn with their layer
        calls[i].with_it += names
        calls[i].out = after_them
    return Trace(_fold(calls), tuple(x.shape[1:]), sum(p.numel() for p in model.parameters()), error, stopped_in)


def _fold(calls: list[Layer]) -> list[Layer]:
    """Draw activations, normalization and the like with the box they follow, and say which boxes feed each box."""
    out: list[Layer] = []
    box_of = {-1: -1}     # call -> its box; the scene is -1
    for i, c in enumerate(calls):
        feeds = {box_of.get(f) for f in c.fed} - {None}
        if (out and isinstance(c.module, _WITH_BEFORE) and c.done and out[-1].done
                and (not c.fed or feeds == {len(out) - 1})):
            last = out[-1]
            last.with_it += [type(c.module).__name__] + c.with_it
            last.params += c.params
            if not isinstance(c.module, nn.Flatten):   # a flatten only reshapes: the box keeps its maps
                last.shape_out, last.out = c.shape_out, c.out
        else:
            c.fed = tuple(sorted(feeds))
            out.append(c)
        box_of[i] = len(out) - 1
    return out


# --- the picture -------------------------------------------------------------------------------------------------

def _whole(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def _shape_text(shape: tuple | None) -> str:
    return "?" if shape is None else "(" + ", ".join(str(s) for s in shape) + ")"


def _mask(a: np.ndarray) -> str:
    """A grey picture of a feature map, light where it is highest, used as a mask over a square in ink, so the
    picture follows the theme. At its own size up to 64 by 64, drawn pixelated: a map of 16 by 16 shows its places."""
    a = a[::max(1, a.shape[0] // 64), ::max(1, a.shape[1] // 64)]   # at most about 64 by 64: the picture is 64 px
    lo, hi = float(a.min()), float(a.max())
    g = np.zeros_like(a, dtype=np.float32) if not hi - lo > 1e-9 else (a - lo) / (hi - lo)
    from PIL import Image
    img = Image.fromarray((np.nan_to_num(g) * 255).astype(np.uint8), "L")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _masked(mid: str, href: str, width: int, height: int) -> str:
    """A small picture in ink through a mask, on the inset: a feature map, or a row of values."""
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true" '
            f'style="display:block;width:{width}px;height:{height}px;max-width:none">'
            f'<defs><mask id="{mid}" maskUnits="userSpaceOnUse" x="0" y="0" width="{width}" height="{height}">'
            f'<image href="{href}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" '
            f'style="image-rendering:pixelated"/></mask></defs>'
            f'<rect width="{width}" height="{height}" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>'
            f'<rect width="{width}" height="{height}" mask="url(#{mid})" style="fill:var(--ea-ink)"/></svg>')


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _settings(module: nn.Module) -> str:
    """The stride and the dilation of a convolution when they are not 1, as PyTorch's arguments: "(dilation=2)"."""
    if not isinstance(module, nn.modules.conv._ConvNd):   # every convolution, 1-D to 3-D, transposed too
        return ""
    said = []
    for name in ("stride", "dilation"):
        v = getattr(module, name)
        if any(x != 1 for x in v):
            said.append(f"{name}={v[0] if len(set(v)) == 1 else tuple(v)}")
    return f"({', '.join(said)})" if said else ""


def _count(shape: tuple) -> int:
    """The number above a box: the channels of maps (C, H, W), the features of a vector (F,) or of every position of a
    sequence (T, F)."""
    return shape[-1] if len(shape) == 2 else shape[0]


def _diagram(tr: Trace, lang: str, chosen: int | None, alt: str, scores: bool) -> tuple[str, bool, bool]:
    """The row of boxes as SVG; and whether it has a dotted edge and an arc, for the key under it."""
    ink, ink2, ink3 = "var(--ea-ink)", "var(--ea-ink-2)", "var(--ea-ink-3)"
    layers = tr.layers
    sides = [s[1] for s in [tr.shape_in] + [l.shape_out for l in layers] if s and len(s) == 3 and s[1]]
    side0 = max(sides, default=1)      # the largest maps sit on top, also when a decoder makes them larger
    deepest = max([math.log2(side0 / s) for s in sides] + [0])

    def level(shape) -> float:
        if shape and len(shape) == 3 and shape[1]:
            return math.log2(side0 / shape[1])
        # a vector of features, a sequence, or nothing: below the smallest map; on top when there are no maps
        return deepest + 1.4 if sides else 0.0

    def size(shape) -> tuple[float, float]:
        if shape and len(shape) == 3:
            return 4 + 2.2 * math.log2(max(shape[0], 1)), 8 + 40 * shape[1] / side0
        if shape and len(shape) in (1, 2):   # a vector of features, or a sequence of them: its features count
            return 6, 8 + 4 * math.log2(max(_count(shape), 1))
        return 8, 12

    labels = [_whole(_count(l.shape_out), lang) if l.done and l.shape_out else ("?" if l.done else "") for l in layers]
    label0 = _whole(_count(tr.shape_in), lang) if tr.shape_in else ""
    step = max(30, 7.2 * max(map(len, labels + [label0])) + 8)   # room for the widest number of channels
    arcs = any(j != i - 1 for i, l in enumerate(layers) for j in l.fed)
    xs = [step * (i + 1) + 40 for i in range(len(layers))]
    levels = [level(l.shape_out if l.done else l.shape_in) for l in layers]
    hs = [size(l.shape_out)[1] if l.done else 20 for l in layers]
    # room above the boxes on top for their channels, and for the arcs: the tallest box of the top row decides
    rows = [(level(tr.shape_in), size(tr.shape_in)[1])] + list(zip(levels, hs))
    first = min(lv for lv, _ in rows)
    tallest = max(h for lv, h in rows if lv == first)
    top = max(46, tallest / 2 + 22) + (22 if arcs else 0)
    ys = [top + _LEVEL * lv for lv in levels]
    parts = []
    w0, h0 = size(tr.shape_in)
    y0 = top + _LEVEL * level(tr.shape_in)
    parts.append(f'<rect x="{40 - w0 / 2:.1f}" y="{y0 - h0 / 2:.1f}" width="{w0:.1f}" height="{h0:.1f}" rx="2" '
                 f'style="fill:var(--ea-data-wash);stroke:var(--ea-data);stroke-width:1.2"/>'
                 f'<text x="40" y="{y0 - h0 / 2 - 5:.1f}" style="font-size:12px;fill:{ink2};text-anchor:middle">'
                 f'{label0}</text>')
    at = {-1: (40, y0, h0)}
    bottom = y0 + h0 / 2
    resized = False
    for i, (l, x, y, h) in enumerate(zip(layers, xs, ys, hs)):
        at[i] = (x, y, h)
        w = size(l.shape_out)[0]
        if not l.done:              # the layer the pass stopped in
            parts.append(f'<rect x="{x - 6}" y="{y - 10:.1f}" width="12" height="20" rx="2" '
                         f'style="fill:none;stroke:{ink};stroke-width:1.5;stroke-dasharray:2 2"/>'
                         f'<text x="{x}" y="{y + 5:.1f}" style="font-size:13px;font-weight:600;fill:{ink};'
                         f'text-anchor:middle">✕</text>')
        else:
            role = "pred" if scores and i == len(layers) - 1 and tr.error is None else "hidden"
            resize = isinstance(l.module, _RESIZE)
            resized |= resize
            style = (f"fill:var(--ea-{role}-wash);stroke:var(--ea-{role});stroke-width:{2.6 if i == chosen else 1.2}"
                     + (";stroke-dasharray:2 2" if resize else ""))
            parts.append(f'<rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="2" '
                         f'style="{style}"/>'
                         f'<text x="{x}" y="{y - h / 2 - 5:.1f}" style="font-size:12px;fill:{ink2};'
                         f'text-anchor:middle">{labels[i]}</text>')
        # a line from the box before, when its output goes in (or when nobody knows where the input came from)
        if not l.fed or i - 1 in l.fed:
            px, py, _ = at[i - 1]
            parts.append(f'<line x1="{px + 5:.1f}" y1="{py:.1f}" x2="{x - 5:.1f}" y2="{y:.1f}" '
                         f'style="stroke:{ink3};stroke-width:1.2"/>')
        # an arc from every earlier box whose output goes in too, over the boxes between
        for j in l.fed:
            if j == i - 1 or j not in at:
                continue
            jx, jy, jh = at[j]
            over = min(at[k][1] - at[k][2] / 2 for k in range(j, i + 1))
            arc = over - 26
            parts.append(f'<path d="M{jx} {jy - jh / 2 - 18:.1f} C{jx} {arc - 14:.1f} {x} {arc - 14:.1f} {x} '
                         f'{y - h / 2 - 18:.1f}" style="fill:none;stroke:{ink3};stroke-width:1.4;'
                         f'stroke-dasharray:4 3"/>')
        if i == chosen:
            parts.append(f'<path d="M{x - 5} {y + h / 2 + 13:.1f} l5 -7 l5 7 Z" style="fill:{ink}"/>')
        bottom = max(bottom, y + h / 2 + 13)
    width = (xs[-1] if xs else 40) + 40
    height = bottom + 6
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}" role="img" '
            f'style="display:block;width:{width:.0f}px;height:{height:.0f}px;max-width:none">'
            f'<title>{_esc(alt)}</title>{"".join(parts)}</svg>', resized, arcs)


def picture(tr: Trace, lang: str, chosen: int | None = None, uid: str = "nv", scores: bool = True) -> str:
    """The network as boxes, from left to right in the order the scene goes through them, and the chosen box with its
    feature maps. `uid` keeps the masks of two pictures on one page apart. With `scores`, the last box gives the
    network's scores and has the colour of a prediction; a block of a network (scores=False) does not."""
    layers = tr.layers
    count = course.phrase("net-one-layer", lang) if len(layers) == 1 else course.phrase("net-layers", lang, n=len(layers))
    title = course.phrase("net-title", lang, layers=count, params=_whole(tr.params, lang))
    svg, resized, arcs = _diagram(tr, lang, chosen, title + (". " + tr.error if tr.error else ""), scores)
    text = "font-size:{}px;color:var(--ea-{});margin:{}"
    out = [f'<div style="overflow-x:auto;max-width:100%">{svg}</div>']
    if tr.error:
        stuck = next((l for l in layers if not l.done), None)
        where = (course.phrase("net-stops-in", lang, name=f"{stuck.name} ({type(stuck.module).__name__})") if stuck
                 else course.phrase("net-stops-after", lang, name=layers[-1].name) if layers
                 else course.phrase("net-stops-before", lang))
        out.append(f'<div style="{text.format(14, "ink", "0.6rem 0 0")};font-weight:600">{_esc(where)}</div>'
                   f'<div style="{text.format(13, "ink-2", "0.2rem 0 0")};font-family:var(--ea-font-code);'
                   f'overflow-wrap:anywhere">{_esc(tr.error)}</div>')
    if chosen is not None and 0 <= chosen < len(layers) and layers[chosen].out is not None:
        out += _chosen(layers[chosen], lang, f"{uid}-{chosen}")
    key = [course.phrase("net-key", lang)] + ([course.phrase("net-key-resize", lang)] if resized else []) \
        + ([course.phrase("net-key-arc", lang)] if arcs else [])
    out.append(f'<div style="{text.format(12, "ink-2", "0.6rem 0 0")}">{"<br>".join(_esc(k) for k in key)}</div>')
    body = f'<div class="ea-netview" style="font-family:var(--ea-font-ui);line-height:1.45">{"".join(out)}</div>'
    # framed as every picture of the learner's work: the title above it
    return yours.figure(title, body, alt=title + (". " + tr.error if tr.error else ""))


def _chosen(l: Layer, lang: str, uid: str) -> list[str]:
    """The chosen layer under the network: its line (name, kind, shapes, parameters) and its first feature maps, or its
    first values when it gives a vector, with what they are. `uid` keeps the masks apart on one page."""
    text = "font-size:{}px;color:var(--ea-{});margin:{}"
    kind = " + ".join([type(l.module).__name__ + _settings(l.module)] + l.with_it)
    line = course.phrase("net-layer", lang, name=l.name, kind=kind, shape_in=_shape_text(l.shape_in),
                         shape_out=_shape_text(l.shape_out), params=_whole(l.params, lang))
    out = [f'<div style="{text.format(14, "ink", "0.7rem 0 0.4rem")}">{_esc(line)}</div>']
    a = l.out.numpy()
    total = l.shape_out[0] if l.shape_out else a.shape[0]   # a layer kept for a rewind holds only the maps shown
    if a.ndim == 3:
        n = min(_SHOWN_MAPS, a.shape[0])
        cells = "".join(
            f'<div style="{text.format(12, "ink-2", "0")};text-align:center">'
            f'{_masked(f"{uid}-{k}", _mask(a[k]), 64, 64)}'
            f'{_esc(course.phrase("net-channel", lang, k=k))}</div>' for k in range(n))
        out.append(f'<div style="display:flex;flex-wrap:wrap;gap:0.3rem 0.4rem">{cells}</div>')
        said = (course.phrase("net-maps-all", lang, n=n) if n == total
                else course.phrase("net-maps-some", lang, n=n, total=_whole(total, lang)))
    else:
        v = a.reshape(-1)
        n = min(_SHOWN_VALUES, v.shape[0])
        total = math.prod(l.shape_out) if l.shape_out else v.shape[0]
        out.append(f'<div style="overflow-x:auto;max-width:100%">'
                   f'{_masked(f"{uid}-v", _mask(v[None, :n]), n * 6, 16)}</div>')
        said = (course.phrase("net-values-all", lang, n=n) if n == total
                else course.phrase("net-values-some", lang, n=n, total=_whole(total, lang)))
    out.append(f'<div style="{text.format(12, "ink-2", "0.3rem 0 0")}">{_esc(said)}</div>')
    return out


@contextlib.contextmanager
def looking(model: nn.Module):
    """Look at a model halfway through its training, as a picture beside the chart does (decision 1.17): in eval mode
    and without a graph, and afterwards every module back in its own mode, also when the look stops with an error. So
    looking takes no step and changes no running statistics, and a random number the learner's network draws while
    it is looked at, such as with F.dropout, which does not follow eval mode, is drawn again in the run: the random
    state is put back. So the run stays the same."""
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            yield model
    finally:
        for m, mode in modes:
            m.training = mode


def last_maps(tr: Trace) -> int | None:
    """The last layer before the scores that gives feature maps, such as the last block on the way up of a U-Net:
    the layer a demo shows while its network trains. None when the pass stopped before there is one."""
    for k in range(len(tr.layers) - 2, -1, -1):
        if tr.layers[k].out is not None and tr.layers[k].out.dim() == 3:
            return k
    return None


def kept(l: Layer) -> Layer:
    """A copy of one layer of a trace with only the feature maps (or values) its picture shows, for a rewind through
    a run: a layer of every epoch, kept small."""
    a = l.out
    if a is not None:
        a = a[:_SHOWN_MAPS].clone() if a.dim() == 3 else a.reshape(-1)[:_SHOWN_VALUES].clone()
    return Layer(l.name, l.module, l.shape_in, l.shape_out, a, l.params, list(l.with_it), l.fed, l.joined, l.done)


def layer_picture(l: Layer, title: str, lang: str, uid: str = "nv-layer") -> str:
    """One layer of the network without the boxes of the whole network: its line and its first feature maps, under
    `title`. Beside the chart of a run while it trains (decision 1.17), and in a rewind through that run."""
    body = f'<div class="ea-netview" style="font-family:var(--ea-font-ui);line-height:1.45">{"".join(_chosen(l, lang, uid))}</div>'
    return yours.figure(title, body, alt=title)


def watched(model: nn.Module, x: torch.Tensor, name: str | None = None) -> Layer | None:
    """The layer a demo watches while its network trains (decision 1.17), on the input x, kept small: the first layer
    with that name that gives feature maps, or when there is none (a learner may name their layers otherwise, or their
    network may stop in it), the last that gives feature maps before the scores (last_maps). None when there is no
    such layer either. It looks as `looking` does, so watching leaves the run as it is, and it never stops the run:
    the picture is the course's own."""
    try:
        with torch.random.fork_rng(devices=[]):
            net = trace(model, x)
        k = next((i for i, l in enumerate(net.layers) if l.name == name and l.out is not None and l.out.dim() == 3),
                 None) if name is not None else None
        k = last_maps(net) if k is None else k
        return None if k is None else kept(net.layers[k])
    except Timeout:   # the demo's time limit
        raise
    except Exception:
        return None


def moment(picture: str, l: Layer | None, title: str, lang: str, uid: str, **fields) -> list[str]:
    """What a demo shows of its network at one moment of a run (decision 1.17): `picture`, what the network makes,
    and under it the layer it watches (None: no layer), under `title` with {name} the layer's name and `fields`
    filled in. `uid` keeps the masks of the layer apart from those of another picture on the page."""
    if l is None or l.out is None:
        return [picture]
    return [picture, layer_picture(l, title.format(name=l.name, **fields), lang, uid)]


def network(name: str, make_input, *args, chosen: int | None = None, scores: bool = True):
    """The picture of an exercise whose answer is a network: the learner's class `name`, made with `args` and
    redrawn from one pass of make_input() every time the check runs, also when it stops halfway (`chosen` and
    `scores` as for picture). Nothing when the code does not run or makes no such class. Seeded, so the same code
    draws the same picture, without touching the learner's own random numbers."""
    def draw(seen) -> str:
        mine = seen.mine
        cls = mine.names.get(name) if mine else None
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            return ""
        # a print or a warning in their forward goes nowhere: the check says what there is to say
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(0)
            try:
                model = cls(*args)
            except Timeout:
                raise
            except Exception:   # their __init__ stops: there is no network to draw, the check says why
                return ""
            return picture(trace(model, make_input()), seen.lang, chosen, uid=f"nv-{name}", scores=scores)
    return draw
