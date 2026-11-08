# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("num_classes", "out_channels"),
    [(3, 4), (0, 5)],
)
@pytest.mark.parametrize(
    "num_key_value_heads_for_query",
    [
        pytest.param(None, id="mha"),
        pytest.param(1, id="mqa"),
    ],
)
def test_icl_block(
    device: torch.device,
    num_classes: int,
    out_channels: int,
    num_key_value_heads_for_query: int | None,
) -> None:
    block = ICLBlock(
        num_classes=num_classes,
        out_channels=out_channels,
        channels=8,
        num_layers=3,
        num_heads=2,
        device=device,
        num_key_value_heads_for_query=num_key_value_heads_for_query,
    )
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(2, 5, 8, device=device)
    if num_classes == 0:
        y = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]],
            device=device,
        )
    else:
        y = torch.randint(num_classes, (2, 3), device=device)

    with torch.no_grad():
        out = block(x.clone(), y)

    assert out.size() == (2, 2, out_channels)
    assert out.dtype == x.dtype
    assert out.device == device

    if num_key_value_heads_for_query is not None:
        mha = ICLBlock(
            num_classes=num_classes,
            out_channels=out_channels,
            channels=8,
            num_layers=3,
            num_heads=2,
            device=device,
        )
        mha.load_state_dict(block.state_dict())
        with torch.no_grad():
            assert not torch.allclose(mha(x.clone(), y), out)

    cache = Cache()
    with torch.no_grad():
        fit_out = block(x[..., :3, :].clone(), y, cache=cache)
        replayed = block(
            x[..., 3:, :].clone(),
            y[..., :0],
            cache=cache.freeze(),
        )

    assert fit_out.size() == (2, 0, out_channels)
    assert cache.size() > 0
    torch.testing.assert_close(replayed, out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("num_classes", [0, 3])
@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("tile_size", [1, 2, 8])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tiled_fit_matches_full_context(
    num_classes: int,
    batch_shape: tuple[int, ...],
    tile_size: int,
    device: str,
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    block = ICLBlock(
        num_classes=num_classes,
        out_channels=5,
        channels=8,
        num_layers=3,
        num_heads=2,
        device=device,
    ).eval()
    tiled = ICLBlock(
        num_classes=num_classes,
        out_channels=5,
        channels=8,
        num_layers=3,
        num_heads=2,
        fit_context_query_tile_size=tile_size,
        device=device,
    ).eval()
    tiled.load_state_dict(block.state_dict())
    x = torch.randn(*batch_shape, 7, 8, device=device)
    y = (
        torch.randint(3, (*batch_shape, 5), device=device)
        if num_classes
        else torch.randn(*batch_shape, 5, device=device)
    )
    full_cache = Cache()
    tiled_cache = Cache()
    with torch.inference_mode():
        full_fit = block(x[..., :5, :].clone(), y, cache=full_cache)
        tiled_fit = tiled(x[..., :5, :].clone(), y, cache=tiled_cache)
        full_prediction = block(
            x[..., 5:, :].clone(), y[..., :0], cache=full_cache.freeze()
        )
        tiled_prediction = tiled(
            x[..., 5:, :].clone(), y[..., :0], cache=tiled_cache.freeze()
        )

    assert full_fit.shape == tiled_fit.shape == (*batch_shape, 0, 5)
    assert full_cache.keys() == tiled_cache.keys()
    for key in full_cache:
        full_entry = full_cache[key]
        tiled_entry = tiled_cache[key]
        assert full_entry.key.shape == tiled_entry.key.shape
        assert full_entry.value.shape == tiled_entry.value.shape
        torch.testing.assert_close(tiled_entry.key, full_entry.key)
        torch.testing.assert_close(tiled_entry.value, full_entry.value)
    torch.testing.assert_close(tiled_prediction, full_prediction)


@pytest.mark.parametrize(
    ("budget_bytes", "tile_size", "query_lengths"),
    [
        (None, None, [5]),
        (160, None, [2, 2, 1]),
        (160, 1, [1, 1, 1, 1, 1]),
    ],
)
def test_fit_attention_budget_selects_tile_size(
    budget_bytes: int | None,
    tile_size: int | None,
    query_lengths: list[int],
) -> None:
    block = ICLBlock(
        num_classes=0,
        out_channels=5,
        channels=8,
        num_layers=2,
        num_heads=2,
        fit_context_query_tile_size=tile_size,
        fit_context_attention_budget_bytes=budget_bytes,
    ).eval()
    lengths: list[int] = []
    handle = block.layers[0].attn.sdpa.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(kwargs["query"].size(-3)),
        with_kwargs=True,
    )
    with torch.inference_mode():
        block(torch.randn(2, 5, 8), torch.randn(2, 5), cache=Cache())
    handle.remove()
    assert lengths == query_lengths


def test_fit_attention_budget_too_small() -> None:
    block = ICLBlock(
        num_classes=0,
        out_channels=5,
        channels=8,
        num_layers=2,
        num_heads=2,
        fit_context_attention_budget_bytes=79,
    ).eval()
    with torch.inference_mode(), pytest.raises(ValueError, match="one context"):
        block(torch.randn(2, 5, 8), torch.randn(2, 5), cache=Cache())


def test_fit_attention_budget_caps_large_tiles() -> None:
    block = ICLBlock(
        num_classes=0,
        out_channels=5,
        channels=8,
        num_layers=2,
        num_heads=2,
        fit_context_attention_budget_bytes=2 * 600 * 4 * 550,
    ).eval()
    lengths: list[int] = []
    handle = block.layers[0].attn.sdpa.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(kwargs["query"].size(-3)),
        with_kwargs=True,
    )
    with torch.inference_mode():
        block(torch.randn(600, 8), torch.randn(600), cache=Cache())
    handle.remove()
    assert lengths == [512, 88]
