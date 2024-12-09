# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib.ticker import MultipleLocator


def spec_to_figure(spec, vmin=None, vmax=None, title=None):
    if isinstance(spec, paddle.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt, title=None):
    if isinstance(dur_gt, paddle.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, paddle.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    width = max(12, min(48, len(txt) // 2))
    fig = plt.figure(figsize=(width, 8))
    plt.vlines(dur_pred, 12, 22, colors="r", label="pred")
    plt.vlines(dur_gt, 0, 10, colors="b", label="gt")
    for i in range(len(txt)):
        shift = i % 8 + 1
        plt.text(
            (dur_pred[i - 1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2,
            12 + shift,
            txt[i],
            size=16,
            horizontalalignment="center",
        )
        plt.text(
            (dur_gt[i - 1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2,
            shift,
            txt[i],
            size=16,
            horizontalalignment="center",
        )
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color="black", linewidth=2, linestyle=":")
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def pitch_note_to_figure(pitch_gt, pitch_pred=None, note_midi=None, note_dur=None, note_rest=None, title=None):
    if isinstance(pitch_gt, paddle.Tensor):
        pitch_gt = pitch_gt.cpu().numpy()
    if isinstance(pitch_pred, paddle.Tensor):
        pitch_pred = pitch_pred.cpu().numpy()
    if isinstance(note_midi, paddle.Tensor):
        note_midi = note_midi.cpu().numpy()
    if isinstance(note_dur, paddle.Tensor):
        note_dur = note_dur.cpu().numpy()
    if isinstance(note_rest, paddle.Tensor):
        note_rest = note_rest.cpu().numpy()
    fig = plt.figure()
    if note_midi is not None and note_dur is not None:
        note_dur_acc = np.cumsum(note_dur)
        if note_rest is None:
            note_rest = np.zeros_like(note_midi, dtype=np.bool_)
        for i in range(len(note_midi)):
            plt.gca().add_patch(
                plt.Rectangle(
                    xy=(note_dur_acc[i - 1] if i > 0 else 0, note_midi[i] - 0.5),
                    width=note_dur[i],
                    height=1,
                    edgecolor="grey",
                    fill=False,
                    linewidth=1.5,
                    linestyle="--" if note_rest[i] else "-",
                )
            )
    plt.plot(pitch_gt, color="b", label="gt")
    if pitch_pred is not None:
        plt.plot(pitch_pred, color="r", label="pred")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def curve_to_figure(curve_gt, curve_pred=None, curve_base=None, grid=None, title=None):
    if isinstance(curve_gt, paddle.Tensor):
        curve_gt = curve_gt.cpu().numpy()
    if isinstance(curve_pred, paddle.Tensor):
        curve_pred = curve_pred.cpu().numpy()
    if isinstance(curve_base, paddle.Tensor):
        curve_base = curve_base.cpu().numpy()
    fig = plt.figure()
    if curve_base is not None:
        plt.plot(curve_base, color="g", label="base")
    plt.plot(curve_gt, color="b", label="gt")
    if curve_pred is not None:
        plt.plot(curve_pred, color="r", label="pred")
    if grid is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(grid))
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def distribution_to_figure(title, x_label, y_label, items: list, values: list, zoom=0.8):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha="center", va="bottom", fontsize=15)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    return fig
