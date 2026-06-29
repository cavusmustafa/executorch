#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenVINO weight-compression and partitioner helpers for the Gemma 4 export.

Kept out of ``examples/models/gemma4/export_gemma4.py`` so the model export
carries only the thin ``--backend openvino`` plumbing and imports these when the
OpenVINO backend is selected.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def apply_ov_weight_compression(
    model, example_inputs, dynamic_shapes, ov_quantize: str, group_size: int
):
    """Data-free NNCF weight compression (INT4/INT8) on the captured FX graph.

    Captures the model with torch.export, runs nncf compress_pt2e (no calibration
    dataset — pure weight compression), and re-exports the compressed graph so the
    OpenVINO partitioner sees the quantized weights.
    """
    import nncf  # noqa: F401
    from executorch.backends.openvino.quantizer import (
        OpenVINOQuantizer,
        QuantizationMode,
    )
    from nncf.experimental.torch.fx import compress_pt2e

    if ov_quantize == "4wo":
        quantizer = OpenVINOQuantizer(
            mode=QuantizationMode.INT4WO_SYM, group_size=group_size, ratio=1
        )
    elif ov_quantize == "8wo":
        quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT8WO_ASYM, group_size=-1)
    else:
        raise ValueError(f"Unsupported ov_quantize mode: {ov_quantize}")

    logger.info(f"Applying NNCF weight compression (openvino_{ov_quantize})...")
    captured = torch.export.export(
        model,
        (example_inputs[0],),
        kwargs={"input_pos": example_inputs[1], "inputs_embeds": example_inputs[2]},
        dynamic_shapes=dynamic_shapes,
    ).module()

    compressed = compress_pt2e(captured, quantizer=quantizer, dataset=None)

    return torch.export.export(
        compressed,
        (example_inputs[0],),
        kwargs={"input_pos": example_inputs[1], "inputs_embeds": example_inputs[2]},
        dynamic_shapes=dynamic_shapes,
    )


def build_openvino_partitioners(
    include_audio: bool,
    include_vision: bool,
    device: str,
) -> dict:
    """Build per-method OpenVINO partitioner lists targeting `device`."""
    from executorch.backends.openvino.partitioner import OpenvinoPartitioner
    from executorch.exir.backend.backend_details import CompileSpec

    def _ov() -> list:
        return [OpenvinoPartitioner([CompileSpec("device", device.encode())])]

    partitioners = {}
    if include_audio:
        partitioners["speech_transform"] = _ov()
        partitioners["audio_encoder"] = _ov()
    if include_vision:
        partitioners["vision_encoder"] = _ov()
    partitioners["text_decoder"] = _ov()
    return partitioners
