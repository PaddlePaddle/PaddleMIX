# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import paddle

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...utils import BaseOutput
from .camera import create_pan_cameras


def sample_pmf(pmf: paddle.Tensor, n_samples: int) -> paddle.Tensor:
    """
    Sample from the given discrete probability distribution with replacement.

    The i-th bin is assumed to have mass pmf[i].

    Args:
        pmf: [batch_size, *shape, n_samples, 1] where (pmf.sum(axis=-2) == 1).all()
        n_samples: number of samples

    Return:
        indices sampled with replacement
    """
    *shape, support_size, last_dim = pmf.shape
    assert last_dim == 1
    cdf = paddle.cumsum(x=pmf.reshape([-1, support_size]), axis=1)
    inds = paddle.searchsorted(
        sorted_sequence=cdf.cast(paddle.float32),
        values=paddle.rand(shape=[cdf.shape[0], n_samples]).cast(paddle.float32),
    )
    return inds.reshape([*shape, n_samples, 1]).clip(min=0, max=support_size - 1)


def posenc_nerf(x: paddle.Tensor, min_deg: int = 0, max_deg: int = 15) -> paddle.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://arxiv.org/pdf/2210.04628.pdf
    """
    if min_deg == max_deg:
        return x
    scales = 2.0 ** paddle.arange(start=min_deg, end=max_deg).astype(x.dtype)
    *shape, dim = x.shape
    xb = (x.reshape([-1, 1, dim]) * scales.reshape([1, -1, 1])).reshape([*shape, -1])
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = paddle.concat(x=[xb, xb + math.pi / 2.0], axis=-1).sin()
    return paddle.concat(x=[x, emb], axis=-1)


def encode_position(position):
    return posenc_nerf(position, min_deg=0, max_deg=15)


def encode_direction(position, direction=None):
    if direction is None:
        return paddle.zeros_like(x=posenc_nerf(position, min_deg=0, max_deg=8))
    else:
        return posenc_nerf(direction, min_deg=0, max_deg=8)


def _sanitize_name(x: str) -> str:
    return x.replace(".", "__")


def integrate_samples(volume_range, ts, density, channels):
    """
    Function integrating the model output.

    Args:
        volume_range: Specifies the integral range [t0, t1]
        ts: timesteps
        density: paddle.Tensor [batch_size, *shape, n_samples, 1]
        channels: paddle.Tensor [batch_size, *shape, n_samples, n_channels]
    returns:
        channels: integrated rgb output weights: paddle.Tensor [batch_size, *shape, n_samples, 1] (density
        *transmittance)[i] weight for each rgb output at [..., i, :]. transmittance: transmittance of this volume
    )
    """
    # 1. Calculate the weights
    _, _, dt = volume_range.partition(ts)
    ddensity = density * dt
    mass = paddle.cumsum(x=ddensity, axis=-2)
    transmittance = paddle.exp(x=-mass[(...), (-1), :])
    alphas = 1.0 - paddle.exp(x=-ddensity)
    Ts = paddle.exp(x=paddle.concat(x=[paddle.zeros_like(x=mass[(...), :1, :]), -mass[(...), :-1, :]], axis=-2))
    # This is the probability of light hitting and reflecting off of
    # something at depth [..., i, :].
    weights = alphas * Ts

    # 2. Integrate channels
    channels = paddle.sum(x=channels * weights, axis=-2)
    return channels, weights, transmittance


def volume_query_points(volume, grid_size):
    indices = paddle.arange(end=grid_size**3)
    zs = indices % grid_size
    ys = paddle.trunc(paddle.divide(indices, paddle.to_tensor(grid_size))) % grid_size
    xs = paddle.trunc(paddle.divide(indices, paddle.to_tensor(grid_size**2))) % grid_size
    combined = paddle.stack(x=[xs, ys, zs], axis=1)
    return combined.astype(dtype="float32") / (grid_size - 1) * (volume.bbox_max - volume.bbox_min) + volume.bbox_min


def _convert_srgb_to_linear(u: paddle.Tensor):
    return paddle.where(condition=u <= 0.04045, x=u / 12.92, y=((u + 0.055) / 1.055) ** 2.4)


def _create_flat_edge_indices(flat_cube_indices: paddle.Tensor, grid_size: Tuple[int, int, int]):
    num_xs = (grid_size[0] - 1) * grid_size[1] * grid_size[2]
    y_offset = num_xs
    num_ys = grid_size[0] * (grid_size[1] - 1) * grid_size[2]
    z_offset = num_xs + num_ys
    return paddle.stack(
        x=[
            flat_cube_indices[:, (0)] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, (1)] * grid_size[2]
            + flat_cube_indices[:, (2)],
            flat_cube_indices[:, (0)] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, (1)] + 1) * grid_size[2]
            + flat_cube_indices[:, (2)],
            flat_cube_indices[:, (0)] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, (1)] * grid_size[2]
            + flat_cube_indices[:, (2)]
            + 1,
            flat_cube_indices[:, (0)] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, (1)] + 1) * grid_size[2]
            + flat_cube_indices[:, (2)]
            + 1,
            y_offset
            + flat_cube_indices[:, (0)] * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, (1)] * grid_size[2]
            + flat_cube_indices[:, (2)],
            y_offset
            + (flat_cube_indices[:, (0)] + 1) * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, (1)] * grid_size[2]
            + flat_cube_indices[:, (2)],
            y_offset
            + flat_cube_indices[:, (0)] * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, (1)] * grid_size[2]
            + flat_cube_indices[:, (2)]
            + 1,
            y_offset
            + (flat_cube_indices[:, (0)] + 1) * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, (1)] * grid_size[2]
            + flat_cube_indices[:, (2)]
            + 1,
            z_offset
            + flat_cube_indices[:, (0)] * grid_size[1] * (grid_size[2] - 1)
            + flat_cube_indices[:, (1)] * (grid_size[2] - 1)
            + flat_cube_indices[:, (2)],
            z_offset
            + (flat_cube_indices[:, (0)] + 1) * grid_size[1] * (grid_size[2] - 1)
            + flat_cube_indices[:, (1)] * (grid_size[2] - 1)
            + flat_cube_indices[:, (2)],
            z_offset
            + flat_cube_indices[:, (0)] * grid_size[1] * (grid_size[2] - 1)
            + (flat_cube_indices[:, (1)] + 1) * (grid_size[2] - 1)
            + flat_cube_indices[:, (2)],
            z_offset
            + (flat_cube_indices[:, (0)] + 1) * grid_size[1] * (grid_size[2] - 1)
            + (flat_cube_indices[:, (1)] + 1) * (grid_size[2] - 1)
            + flat_cube_indices[:, (2)],
        ],
        axis=-1,
    )


class VoidNeRFModel(paddle.nn.Layer):
    """
    Implements the default empty space model where all queries are rendered as background.
    """

    def __init__(self, background, channel_scale=255.0):
        super().__init__()
        out_70 = paddle.create_parameter(
            shape=(paddle.to_tensor(data=np.array(background)).cast(dtype="float32") / channel_scale).shape,
            dtype=(paddle.to_tensor(data=np.array(background)).cast(dtype="float32") / channel_scale).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(data=np.array(background)).cast(dtype="float32") / channel_scale
            ),
        )
        out_70.stop_gradient = not True
        background = out_70
        self.add_parameter("background", background)

    def forward(self, position):
        background = self.background[None]
        shape = position.shape[:-1]
        ones = [1] * (len(shape) - 1)
        n_channels = background.shape[-1]
        background = paddle.broadcast_to(
            x=background.reshape([background.shape[0], *ones, n_channels]), shape=[*shape, n_channels]
        )
        return background


@dataclass
class VolumeRange:
    t0: paddle.Tensor
    t1: paddle.Tensor
    intersected: paddle.Tensor

    def __post_init__(self):
        assert self.t0.shape == self.t1.shape == self.intersected.shape

    def partition(self, ts):
        """
        Partitions t0 and t1 into n_samples intervals.

        Args:
            ts: [batch_size, *shape, n_samples, 1]

        Return:

            lower: [batch_size, *shape, n_samples, 1] upper: [batch_size, *shape, n_samples, 1] delta: [batch_size,
            *shape, n_samples, 1]

        where
            ts \\in [lower, upper] deltas = upper - lower
        """
        mids = (ts[(...), 1:, :] + ts[(...), :-1, :]) * 0.5
        lower = paddle.concat(x=[self.t0[(...), (None), :], mids], axis=-2)
        upper = paddle.concat(x=[mids, self.t1[(...), (None), :]], axis=-2)
        delta = upper - lower
        assert lower.shape == upper.shape == delta.shape == ts.shape
        return lower, upper, delta


class BoundingBoxVolume(paddle.nn.Layer):
    """
    Axis-aligned bounding box defined by the two opposite corners.
    """

    def __init__(self, *, bbox_min, bbox_max, min_dist: float = 0.0, min_t_range: float = 0.001):
        """
        Args:
            bbox_min: the left/bottommost corner of the bounding box
            bbox_max: the other corner of the bounding box
            min_dist: all rays should start at least this distance away from the origin.
        """
        super().__init__()
        self.min_dist = min_dist
        self.min_t_range = min_t_range
        self.bbox_min = paddle.to_tensor(data=bbox_min)
        self.bbox_max = paddle.to_tensor(data=bbox_max)
        self.bbox = paddle.stack(x=[self.bbox_min, self.bbox_max])
        assert self.bbox.shape == [2, 3]
        assert min_dist >= 0.0
        assert min_t_range > 0.0

    def intersect(
        self, origin: paddle.Tensor, direction: paddle.Tensor, t0_lower: Optional[paddle.Tensor] = None, epsilon=1e-06
    ):
        """
        Args:
            origin: [batch_size, *shape, 3]
            direction: [batch_size, *shape, 3]
            t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
            params: Optional meta parameters in case Volume is parametric
            epsilon: to stabilize calculations

        Return:
            A tuple of (t0, t1, intersected) where each has a shape [batch_size, *shape, 1]. If a ray intersects with
            the volume, `o + td` is in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed to
            be on the boundary of the volume.
        """
        batch_size, *shape, _ = origin.shape
        ones = [1] * len(shape)

        bbox = self.bbox.reshape([1, *ones, 2, 3])

        def _safe_divide(a, b, epsilon=1e-06):
            return a / paddle.where(condition=b < 0, x=b - epsilon, y=b + epsilon)

        ts = _safe_divide(bbox - origin[(...), (None), :], direction[(...), (None), :], epsilon=epsilon)
        # Cases to think about:
        #
        #   1. t1 <= t0: the ray does not pass through the AABB.
        #   2. t0 < t1 <= 0: the ray intersects but the BB is behind the origin.
        #   3. t0 <= 0 <= t1: the ray starts from inside the BB
        #   4. 0 <= t0 < t1: the ray is not inside and intersects with the BB twice.
        #
        # 1 and 4 are clearly handled from t0 < t1 below.
        # Making t0 at least min_dist (>= 0) takes care of 2 and 3.
        t0 = ts.min(axis=-2).max(axis=-1, keepdim=True).clip(self.min_dist)
        t1 = ts.max(axis=-2).min(axis=-1, keepdim=True)
        assert t0.shape == t1.shape == [batch_size, *shape, 1]
        if t0_lower is not None:
            assert t0.shape == t0_lower.shape
            t0 = paddle.maximum(x=t0, y=t0_lower)
        intersected = t0 + self.min_t_range < t1
        t0 = paddle.where(condition=intersected, x=t0, y=paddle.zeros_like(x=t0))
        t1 = paddle.where(condition=intersected, x=t1, y=paddle.ones_like(x=t1))
        return VolumeRange(t0=t0, t1=t1, intersected=intersected)


class StratifiedRaySampler(paddle.nn.Layer):
    """
    Instead of fixed intervals, a sample is drawn uniformly at random from each interval.
    """

    def __init__(self, depth_mode: str = "linear"):
        """
        :param depth_mode: linear samples ts linearly in depth. harmonic ensures
            closer points are sampled more densely.
        """
        self.depth_mode = depth_mode
        assert self.depth_mode in ("linear", "geometric", "harmonic")

    def sample(self, t0: paddle.Tensor, t1: paddle.Tensor, n_samples: int, epsilon: float = 0.001) -> paddle.Tensor:
        """
        Args:
            t0: start time has shape [batch_size, *shape, 1]
            t1: finish time has shape [batch_size, *shape, 1]
            n_samples: number of ts to sample
        Return:
            sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        ones = [1] * (len(t0.shape) - 1)
        ts = paddle.linspace(start=0, stop=1, num=n_samples).reshape([*ones, n_samples]).cast(t0.dtype)
        if self.depth_mode == "linear":
            ts = t0 * (1.0 - ts) + t1 * ts
        elif self.depth_mode == "geometric":
            ts = (t0.clip(min=epsilon).log() * (1.0 - ts) + t1.clip(min=epsilon).log() * ts).exp()
        elif self.depth_mode == "harmonic":
            # The original NeRF recommends this interpolation scheme for
            # spherical scenes, but there could be some weird edge cases when
            # the observer crosses from the inner to outer volume.
            ts = 1.0 / (1.0 / t0.clip(min=epsilon) * (1.0 - ts) + 1.0 / t1.clip(min=epsilon) * ts)
        mids = 0.5 * (ts[(...), 1:] + ts[(...), :-1])
        upper = paddle.concat(x=[mids, t1], axis=-1)
        lower = paddle.concat(x=[t0, mids], axis=-1)
        # yiyi notes: add a random seed here for testing, don't forget to remove
        # paddle.seed(seed=0)
        t_rand = paddle.rand(shape=ts.shape, dtype=ts.dtype)
        ts = lower + (upper - lower) * t_rand
        return ts.unsqueeze(axis=-1)


class ImportanceRaySampler(paddle.nn.Layer):
    """
    Given the initial estimate of densities, this samples more from regions/bins expected to have objects.
    """

    def __init__(
        self,
        volume_range: VolumeRange,
        ts: paddle.Tensor,
        weights: paddle.Tensor,
        blur_pool: bool = False,
        alpha: float = 1e-05,
    ):
        """
        Args:
            volume_range: the range in which a ray intersects the given volume.
            ts: earlier samples from the coarse rendering step
            weights: discretized version of density * transmittance
            blur_pool: if true, use 2-tap max + 2-tap blur filter from mip-NeRF.
            alpha: small value to add to weights.
        """
        super().__init__()
        self.volume_range = volume_range
        self.ts = ts.clone().detach()
        self.weights = weights.clone().detach()
        self.blur_pool = blur_pool
        self.alpha = alpha

    @paddle.no_grad()
    def sample(self, t0: paddle.Tensor, t1: paddle.Tensor, n_samples: int) -> paddle.Tensor:
        """
        Args:
            t0: start time has shape [batch_size, *shape, 1]
            t1: finish time has shape [batch_size, *shape, 1]
            n_samples: number of ts to sample
        Return:
            sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        lower, upper, _ = self.volume_range.partition(self.ts)
        batch_size, *shape, n_coarse_samples, _ = self.ts.shape
        weights = self.weights
        if self.blur_pool:
            padded = paddle.concat(x=[weights[(...), :1, :], weights, weights[(...), -1:, :]], axis=-2)
            maxes = paddle.maximum(x=padded[(...), :-1, :], y=padded[(...), 1:, :])
            weights = 0.5 * (maxes[(...), :-1, :] + maxes[(...), 1:, :])
        weights = weights + self.alpha
        pmf = weights / weights.sum(axis=-2, keepdim=True)
        inds = sample_pmf(pmf, n_samples)
        assert inds.shape == [batch_size, *shape, n_samples, 1]
        assert (inds >= 0).astype("bool").all() and (inds < n_coarse_samples).astype("bool").all()
        t_rand = paddle.rand(shape=inds.shape)
        lower_ = paddle.take_along_axis(arr=lower, axis=-2, indices=inds)
        upper_ = paddle.take_along_axis(arr=upper, axis=-2, indices=inds)
        ts = lower_ + (upper_ - lower_) * t_rand
        ts = paddle.sort(x=ts, axis=-2)
        return ts


@dataclass
class MeshDecoderOutput(BaseOutput):
    """
    A 3D triangle mesh with optional data at the vertices and faces.

    Args:
        verts (`paddle.Tensor` of shape `(N, 3)`):
            array of vertext coordinates
        faces (`paddle.Tensor` of shape `(N, 3)`):
            array of triangles, pointing to indices in verts.
        vertext_channels (Dict):
            vertext coordinates for each color channel
    """

    verts: paddle.Tensor
    faces: paddle.Tensor
    vertex_channels: Dict[str, paddle.Tensor]


class MeshDecoder(paddle.nn.Layer):
    """
    Construct meshes from Signed distance functions (SDFs) using marching cubes method
    """

    def __init__(self):
        super().__init__()
        cases = paddle.zeros(shape=[256, 5, 3], dtype="int64")
        masks = paddle.zeros(shape=[256, 5], dtype="bool")
        self.register_buffer(name="cases", tensor=cases)
        self.register_buffer(name="masks", tensor=masks)

    def forward(self, field: paddle.Tensor, min_point: paddle.Tensor, size: paddle.Tensor):
        """
        For a signed distance field, produce a mesh using marching cubes.

        :param field: a 3D tensor of field values, where negative values correspond
                    to the outside of the shape. The dimensions correspond to the x, y, and z directions, respectively.
        :param min_point: a tensor of shape [3] containing the point corresponding
                        to (0, 0, 0) in the field.
        :param size: a tensor of shape [3] containing the per-axis distance from the
                    (0, 0, 0) field corner and the (-1, -1, -1) field corner.
        """
        assert len(field.shape) == 3, "input must be a 3D scalar field"
        cases = self.cases
        masks = self.masks
        min_point = min_point
        size = size
        grid_size = field.shape
        grid_size_tensor = paddle.to_tensor(data=grid_size).cast(size.dtype)

        # Create bitmasks between 0 and 255 (inclusive) indicating the state
        # of the eight corners of each cube.
        bitmasks = (field > 0).cast("uint8")
        bitmasks = bitmasks[:-1, :, :] | bitmasks[1:, :, :] << 1
        bitmasks = bitmasks[:, :-1, :] | bitmasks[:, 1:, :] << 2
        bitmasks = bitmasks[:, :, :-1] | bitmasks[:, :, 1:] << 4
        # Compute corner coordinates across the entire grid.
        corner_coords = paddle.empty(shape=[*grid_size, 3], dtype=field.dtype)
        corner_coords[(range(grid_size[0])), :, :, (0)] = paddle.arange(end=grid_size[0]).astype(field.dtype)[
            :, (None), (None)
        ]
        corner_coords[:, (range(grid_size[1])), :, (1)] = paddle.arange(end=grid_size[1]).astype(field.dtype)[
            :, (None)
        ]
        corner_coords[:, :, (range(grid_size[2])), (2)] = paddle.arange(end=grid_size[2]).astype(field.dtype)
        # Compute all vertices across all edges in the grid, even though we will
        # throw some out later. We have (X-1)*Y*Z + X*(Y-1)*Z + X*Y*(Z-1) vertices.
        # These are all midpoints, and don't account for interpolation (which is
        # done later based on the used edge midpoints).
        edge_midpoints = paddle.concat(
            x=[
                ((corner_coords[:-1] + corner_coords[1:]) / 2).reshape([-1, 3]),
                ((corner_coords[:, :-1] + corner_coords[:, 1:]) / 2).reshape([-1, 3]),
                ((corner_coords[:, :, :-1] + corner_coords[:, :, 1:]) / 2).reshape([-1, 3]),
            ],
            axis=0,
        )
        # Create a flat array of [X, Y, Z] indices for each cube.
        cube_indices = paddle.zeros(shape=[grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1, 3], dtype="int64")
        cube_indices[(range(grid_size[0] - 1)), :, :, (0)] = paddle.arange(end=grid_size[0] - 1)[:, (None), (None)]
        cube_indices[:, (range(grid_size[1] - 1)), :, (1)] = paddle.arange(end=grid_size[1] - 1)[:, (None)]
        cube_indices[:, :, (range(grid_size[2] - 1)), (2)] = paddle.arange(end=grid_size[2] - 1)
        flat_cube_indices = cube_indices.reshape([-1, 3])
        # Create a flat array mapping each cube to 12 global edge indices.
        edge_indices = _create_flat_edge_indices(flat_cube_indices, grid_size)
        # Apply the LUT to figure out the triangles.
        flat_bitmasks = bitmasks.reshape([-1]).astype(
            dtype="int64"
        )  # must cast to long for indexing to believe this not a mask
        local_tris = cases[flat_bitmasks]
        local_masks = masks[flat_bitmasks]
        # Compute the global edge indices for the triangles.
        global_tris = paddle.take_along_axis(
            arr=edge_indices, axis=1, indices=local_tris.reshape([local_tris.shape[0], -1])
        ).reshape(local_tris.shape)
        # Select the used triangles for each cube.
        selected_tris = global_tris.reshape([-1, 3])[
            local_masks.reshape(
                [
                    -1,
                ]
            )
        ]

        # Now we have a bunch of indices into the full list of possible vertices,
        # but we want to reduce this list to only the used vertices.
        used_vertex_indices = paddle.unique(
            x=selected_tris.reshape(
                [
                    -1,
                ]
            )
        )
        used_edge_midpoints = edge_midpoints[used_vertex_indices]
        old_index_to_new_index = paddle.zeros(shape=len(edge_midpoints), dtype="int64")
        old_index_to_new_index[used_vertex_indices] = paddle.arange(end=len(used_vertex_indices)).astype("int64")
        # Rewrite the triangles to use the new indices
        faces = paddle.take_along_axis(
            arr=old_index_to_new_index,
            axis=0,
            indices=selected_tris.reshape(
                [
                    -1,
                ]
            ),
        ).reshape(selected_tris.shape)

        # Compute the actual interpolated coordinates corresponding to edge midpoints.
        v1 = paddle.floor(x=used_edge_midpoints).cast("int64")
        v2 = paddle.ceil(x=used_edge_midpoints).cast("int64")
        s1 = field[v1[:, (0)], v1[:, (1)], v1[:, (2)]]
        s2 = field[v2[:, (0)], v2[:, (1)], v2[:, (2)]]
        p1 = v1.astype(dtype="float32") / (grid_size_tensor - 1) * size + min_point
        p2 = v2.astype(dtype="float32") / (grid_size_tensor - 1) * size + min_point
        # The signs of s1 and s2 should be different. We want to find
        # t such that t*s2 + (1-t)*s1 = 0.
        t = (s1 / (s1 - s2))[:, (None)]
        verts = t * p2 + (1 - t) * p1
        return MeshDecoderOutput(verts=verts, faces=faces, vertex_channels=None)


@dataclass
class MLPNeRFModelOutput(BaseOutput):
    density: paddle.Tensor
    signed_distance: paddle.Tensor
    channels: paddle.Tensor
    ts: paddle.Tensor


class MLPNeRSTFModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        d_hidden: int = 256,
        n_output: int = 12,
        n_hidden_layers: int = 6,
        act_fn: str = "swish",
        insert_direction_at: int = 4,
    ):
        super().__init__()

        # Instantiate the MLP

        # Find out the dimension of encoded position and direction
        dummy = paddle.eye(num_rows=1, num_columns=3)
        d_posenc_pos = encode_position(position=dummy).shape[-1]
        d_posenc_dir = encode_direction(position=dummy).shape[-1]
        mlp_widths = [d_hidden] * n_hidden_layers
        input_widths = [d_posenc_pos] + mlp_widths
        output_widths = mlp_widths + [n_output]
        if insert_direction_at is not None:
            input_widths[insert_direction_at] += d_posenc_dir
        self.mlp = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(in_features=d_in, out_features=d_out)
                for d_in, d_out in zip(input_widths, output_widths)
            ]
        )
        if act_fn == "swish":
            # self.activation = swish
            # yiyi testing:
            self.activation = lambda x: paddle.nn.functional.silu(x=x)
        else:
            raise ValueError(f"Unsupported activation function {act_fn}")
        self.sdf_activation = paddle.tanh
        self.density_activation = paddle.nn.functional.relu
        self.channel_activation = paddle.nn.functional.sigmoid

    def map_indices_to_keys(self, output):
        h_map = {
            "sdf": (0, 1),
            "density_coarse": (1, 2),
            "density_fine": (2, 3),
            "stf": (3, 6),
            "nerf_coarse": (6, 9),
            "nerf_fine": (9, 12),
        }
        mapped_output = {k: output[(...), start:end] for k, (start, end) in h_map.items()}
        return mapped_output

    def forward(self, *, position, direction, ts, nerf_level="coarse", rendering_mode="nerf"):
        h = encode_position(position)
        h_preact = h
        h_directionless = None

        for i, layer in enumerate(self.mlp):
            if i == self.config.insert_direction_at:
                h_directionless = h_preact
                h_direction = encode_direction(position, direction=direction)
                h = paddle.concat(x=[h, h_direction], axis=-1)
            h = layer(h)
            h_preact = h
            if i < len(self.mlp) - 1:
                h = self.activation(h)
        h_final = h
        if h_directionless is None:
            h_directionless = h_preact
        activation = self.map_indices_to_keys(h_final)
        if nerf_level == "coarse":
            h_density = activation["density_coarse"]
        else:
            h_density = activation["density_fine"]
        if rendering_mode == "nerf":
            if nerf_level == "coarse":
                h_channels = activation["nerf_coarse"]
            else:
                h_channels = activation["nerf_fine"]
        elif rendering_mode == "stf":
            h_channels = activation["stf"]
        density = self.density_activation(h_density)
        signed_distance = self.sdf_activation(activation["sdf"])
        channels = self.channel_activation(h_channels)

        # yiyi notes: I think signed_distance is not used
        return MLPNeRFModelOutput(density=density, signed_distance=signed_distance, channels=channels, ts=ts)


class ChannelsProj(paddle.nn.Layer):
    def __init__(self, *, vectors: int, channels: int, d_latent: int):
        super().__init__()
        self.proj = paddle.nn.Linear(in_features=d_latent, out_features=vectors * channels)
        self.norm = paddle.nn.LayerNorm(normalized_shape=channels)
        self.d_latent = d_latent
        self.vectors = vectors
        self.channels = channels

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x_bvd = x
        w_vcd = self.proj.weight.reshape([self.d_latent, self.vectors, self.channels])

        b_vc = self.proj.bias.reshape([1, self.vectors, self.channels])
        h = paddle.einsum("bvd,dvc->bvc", x_bvd, w_vcd.cast(x.dtype))
        h = h.cast(b_vc.dtype)
        h = self.norm(h)
        h = h + b_vc
        return h


class ShapEParamsProjModel(ModelMixin, ConfigMixin):
    """
    project the latent representation of a 3D asset to obtain weights of a multi-layer perceptron (MLP).

    For more details, see the original paper:
    """

    @register_to_config
    def __init__(
        self,
        *,
        param_names: Tuple[str] = (
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = ((256, 93), (256, 256), (256, 256), (256, 256)),
        d_latent: int = 1024
    ):
        super().__init__()

        # check inputs
        if len(param_names) != len(param_shapes):
            raise ValueError("Must provide same number of `param_names` as `param_shapes`")
        self.projections = paddle.nn.LayerDict(sublayers={})
        for k, (vectors, channels) in zip(param_names, param_shapes):
            self.projections[_sanitize_name(k)] = ChannelsProj(vectors=vectors, channels=channels, d_latent=d_latent)

    def forward(self, x: paddle.Tensor):
        out = {}
        start = 0
        for k, shape in zip(self.config.param_names, self.config.param_shapes):
            vectors, _ = shape
            end = start + vectors
            x_bvd = x[:, start:end]
            out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape([len(x), *shape])
            start = end
        return out


class ShapERenderer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        param_names: Tuple[str] = (
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = ((256, 93), (256, 256), (256, 256), (256, 256)),
        d_latent: int = 1024,
        d_hidden: int = 256,
        n_output: int = 12,
        n_hidden_layers: int = 6,
        act_fn: str = "swish",
        insert_direction_at: int = 4,
        background: Tuple[float] = (255.0, 255.0, 255.0)
    ):
        super().__init__()
        self.params_proj = ShapEParamsProjModel(param_names=param_names, param_shapes=param_shapes, d_latent=d_latent)
        self.mlp = MLPNeRSTFModel(d_hidden, n_output, n_hidden_layers, act_fn, insert_direction_at)
        self.void = VoidNeRFModel(background=background, channel_scale=255.0)
        self.volume = BoundingBoxVolume(bbox_max=[1.0, 1.0, 1.0], bbox_min=[-1.0, -1.0, -1.0])
        self.mesh_decoder = MeshDecoder()

    @paddle.no_grad()
    def render_rays(self, rays, sampler, n_samples, prev_model_out=None, render_with_direction=False):
        """
        Perform volumetric rendering over a partition of possible t's in the union of rendering volumes (written below
        with some abuse of notations)

            C(r) := sum(
                transmittance(t[i]) * integrate(
                    lambda t: density(t) * channels(t) * transmittance(t), [t[i], t[i + 1]],
                ) for i in range(len(parts))
            ) + transmittance(t[-1]) * void_model(t[-1]).channels

        where

        1) transmittance(s) := exp(-integrate(density, [t[0], s])) calculates the probability of light passing through
        the volume specified by [t[0], s]. (transmittance of 1 means light can pass freely) 2) density and channels are
        obtained by evaluating the appropriate part.model at time t. 3) [t[i], t[i + 1]] is defined as the range of t
        where the ray intersects (parts[i].volume \\ union(part.volume for part in parts[:i])) at the surface of the
        shell (if bounded). If the ray does not intersect, the integral over this segment is evaluated as 0 and
        transmittance(t[i + 1]) := transmittance(t[i]). 4) The last term is integration to infinity (e.g. [t[-1],
        math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).

        args:
            rays: [batch_size x ... x 2 x 3] origin and direction. sampler: disjoint volume integrals. n_samples:
            number of ts to sample. prev_model_outputs: model outputs from the previous rendering step, including

        :return: A tuple of
            - `channels`
            - A importance samplers for additional fine-grained rendering
            - raw model output
        """
        origin, direction = rays[(...), (0), :], rays[(...), (1), :]

        # Integrate over [t[i], t[i + 1]]

        # 1 Intersect the rays with the current volume and sample ts to integrate along.
        vrange = self.volume.intersect(origin, direction, t0_lower=None)
        ts = sampler.sample(vrange.t0, vrange.t1, n_samples)
        ts = ts.cast(rays.dtype)
        if prev_model_out is not None:
            # Append the previous ts now before fprop because previous
            # rendering used a different model and we can't reuse the output.
            ts = paddle.sort(x=paddle.concat(x=[ts, prev_model_out.ts], axis=-2), axis=-2)
            # paddle.argsort(x=paddle.concat(x=[ts, prev_model_out.ts], axis=-2), axis=-2),
        batch_size, *_shape, _t0_dim = vrange.t0.shape
        _, *ts_shape, _ts_dim = ts.shape
        # 2. Get the points along the ray and query the model
        directions = paddle.broadcast_to(x=direction.unsqueeze(axis=-2), shape=[batch_size, *ts_shape, 3])
        positions = origin.unsqueeze(axis=-2) + ts * directions
        directions = directions.cast(self.mlp.dtype)
        positions = positions.cast(self.mlp.dtype)
        optional_directions = directions if render_with_direction else None

        model_out = self.mlp(
            position=positions,
            direction=optional_directions,
            ts=ts,
            nerf_level="coarse" if prev_model_out is None else "fine",
        )
        # 3. Integrate the model results
        channels, weights, transmittance = integrate_samples(
            vrange, model_out.ts, model_out.density, model_out.channels
        )
        # 4. Clean up results that do not intersect with the volume.
        transmittance = paddle.where(
            condition=vrange.intersected, x=transmittance, y=paddle.ones_like(x=transmittance)
        )
        channels = paddle.where(condition=vrange.intersected, x=channels, y=paddle.zeros_like(x=channels))
        # 5. integration to infinity (e.g. [t[-1], math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).
        channels = channels + transmittance * self.void(origin)
        weighted_sampler = ImportanceRaySampler(vrange, ts=model_out.ts, weights=weights)
        return channels, weighted_sampler, model_out

    @paddle.no_grad()
    def decode_to_image(
        self, latents, size: int = 64, ray_batch_size: int = 4096, n_coarse_samples=64, n_fine_samples=128
    ):
        # project the the parameters from the generated latents
        projected_params = self.params_proj(latents)

        # update the mlp layers of the renderer
        for name, param in self.mlp.state_dict().items():
            if f"nerstf.{name}" in projected_params.keys():
                param.copy_(projected_params[f"nerstf.{name}"].squeeze(axis=0).cast(param.dtype).T, False)

        # create cameras object
        camera = create_pan_cameras(size)
        rays = camera.camera_rays
        n_batches = rays.shape[1] // ray_batch_size
        coarse_sampler = StratifiedRaySampler()
        images = []
        for idx in range(n_batches):
            rays_batch = rays[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]
            # render rays with coarse, stratified samples.
            _, fine_sampler, coarse_model_out = self.render_rays(rays_batch, coarse_sampler, n_coarse_samples)
            # Then, render with additional importance-weighted ray samples.
            channels, _, _ = self.render_rays(
                rays_batch, fine_sampler, n_fine_samples, prev_model_out=coarse_model_out
            )
            images.append(channels)
        images = paddle.concat(x=images, axis=1)
        images = images.reshape([*camera.shape, camera.height, camera.width, -1]).squeeze(axis=0)
        return images

    @paddle.no_grad()
    def decode_to_mesh(
        self, latents, grid_size: int = 128, query_batch_size: int = 4096, texture_channels: Tuple = ("R", "G", "B")
    ):
        # 1. project the the parameters from the generated latents
        projected_params = self.params_proj(latents)

        # 2. update the mlp layers of the renderer
        for name, param in self.mlp.state_dict().items():
            if f"nerstf.{name}" in projected_params.keys():
                param.copy_(projected_params[f"nerstf.{name}"].squeeze(axis=0).cast(param.dtype).T, False)

        # 3. decoding with STF rendering
        # 3.1 query the SDF values at vertices along a regular 128**3 grid

        query_points = volume_query_points(self.volume, grid_size)
        query_positions = query_points[None].tile(repeat_times=[1, 1, 1]).cast(dtype=self.mlp.dtype)
        fields = []
        for idx in range(0, query_positions.shape[1], query_batch_size):
            query_batch = query_positions[:, idx : idx + query_batch_size]
            model_out = self.mlp(
                position=query_batch, direction=None, ts=None, nerf_level="fine", rendering_mode="stf"
            )
            fields.append(model_out.signed_distance)

        # predicted SDF values
        fields = paddle.concat(x=fields, axis=1)
        fields = fields.astype(dtype="float32")
        assert (
            len(fields.shape) == 3 and fields.shape[-1] == 1
        ), f"expected [meta_batch x inner_batch] SDF results, but got {fields.shape}"

        # create grid 128 x 128 x 128
        # - force a negative border around the SDFs to close off all the models.
        fields = fields.reshape([1, *([grid_size] * 3)])
        full_grid = paddle.zeros(shape=[1, grid_size + 2, grid_size + 2, grid_size + 2], dtype=fields.dtype)
        full_grid.fill_(value=-1.0)
        full_grid[:, 1:-1, 1:-1, 1:-1] = fields
        fields = full_grid

        # apply a differentiable implementation of Marching Cubes to construct meshs
        raw_meshes = []
        mesh_mask = []
        for field in fields:
            raw_mesh = self.mesh_decoder(field, self.volume.bbox_min, self.volume.bbox_max - self.volume.bbox_min)
            mesh_mask.append(True)
            raw_meshes.append(raw_mesh)
        mesh_mask = paddle.to_tensor(data=mesh_mask)
        max_vertices = max(len(m.verts) for m in raw_meshes)

        # 3.2. query the texture color head at each vertex of the resulting mesh.
        texture_query_positions = paddle.stack(
            x=[m.verts[paddle.arange(start=0, end=max_vertices) % len(m.verts)] for m in raw_meshes], axis=0
        )
        texture_query_positions = texture_query_positions.cast(dtype=self.mlp.dtype)
        textures = []
        for idx in range(0, texture_query_positions.shape[1], query_batch_size):
            query_batch = texture_query_positions[:, idx : idx + query_batch_size]
            texture_model_out = self.mlp(
                position=query_batch, direction=None, ts=None, nerf_level="fine", rendering_mode="stf"
            )
            textures.append(texture_model_out.channels)

        # predict texture color
        textures = paddle.concat(x=textures, axis=1)
        textures = _convert_srgb_to_linear(textures)
        textures = textures.astype(dtype="float32")

        # 3.3 augument the mesh with texture data
        assert len(textures.shape) == 3 and textures.shape[-1] == len(
            texture_channels
        ), f"expected [meta_batch x inner_batch x texture_channels] field results, but got {textures.shape}"
        for m, texture in zip(raw_meshes, textures):
            texture = texture[: len(m.verts)]
            m.vertex_channels = dict(zip(texture_channels, texture.unbind(axis=-1)))
        return raw_meshes[0]
