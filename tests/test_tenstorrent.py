import pytest, torch
import time
import sys
import os

from functools import partial

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from boltz.model.modules.tenstorrent import (
#from boltz.model.modules.tenstorrent_p150_july21 import (
    PairformerModule,
    DiffusionTransformerModule,
    MSAModule,
    filter_dict
)
from boltz.model.modules.encoders import get_indexing_matrix, single_to_keys
from boltz.model.modules.trunkv2 import MSAModule as MSAModuleTorch
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer as DiffusionTransformerTorch
)
from boltz.model.layers.pairformer import PairformerModule as PairformerModuleTorch

torch.set_grad_enabled(False)
torch.manual_seed(893)

state_dict = torch.load(
    "../cache/boltz2_conf.ckpt", map_location="cpu", mmap=True
)["state_dict"]

import numpy as np
import ttnn
import math
from loguru import logger

def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    num_devices: int | None = None,
    pcc: float | None = None,
    ccc: float | None = None,
    mse: float | None = None,
    relative_rmse: float | None = None,
    shard_dim: int | None = None,
) -> None:
    if isinstance(a, ttnn.Tensor):
        a = to_torch(
            a,
            mesh_device=a.device(),
            dtype=torch.float32,
            shard_dim=shard_dim,
            fix_special_numbers=True,
        )
        if num_devices is not None:
            assert shard_dim == 0
            a = a[0 : a.shape[0] // num_devices, ...]

    if isinstance(b, ttnn.Tensor):
        b = to_torch(
            b,
            mesh_device=b.device(),
            dtype=torch.float32,
            shard_dim=shard_dim,
            fix_special_numbers=True,
        )
        if num_devices is not None:
            assert shard_dim == 0
            b = b[0 : b.shape[0] // num_devices, ...]

    if math.prod(a.shape) != math.prod(b.shape):
        msg = f"incompatible shapes: {a.shape} != {b.shape}"
        raise ValueError(msg)

    if a.shape != b.shape:
        logger.warning(f"shape mismatch: {a.shape} != {b.shape}")

    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)

    cov = torch.cov(torch.stack([a, b])).numpy()

    std_a = math.sqrt(cov[0, 0])
    std_b = math.sqrt(cov[1, 1])
    mean_a = a.mean().item()
    mean_b = b.mean().item()

    pcc_found = cov[0, 1] / (std_a * std_b)
    beta_found = cov[0, 1] / cov[0, 0]
    ccc_found = 2 * pcc_found * std_a * std_b / (std_a**2 + std_b**2 + (mean_a - mean_b) ** 2)
    relative_rmse_found = torch.nn.functional.mse_loss(a, b).sqrt().item() / std_a

    if mse is not None:
        relative_rmse = math.sqrt(mse) / std_a

    logger.info(f"μ₁ = {mean_a:.3g}, μ₂ = {mean_b:.3g}, σ₁ = {std_a:.3g}, σ₂ = {std_b:.3g}")
    logger.info(
        f"PCC = {pcc_found * 100:.4f} %, "
        f"β = {beta_found * 100:.1f} %, "
        f"CCC = {ccc_found * 100:.4f} %, "
        f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} %"
    )

    if pcc is not None and pcc_found < pcc:
        msg = f"PCC = {pcc_found * 100:.4f} % >= {pcc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if ccc is not None and ccc_found < ccc:
        msg = f"CCC = {ccc_found * 100:.4f} % >= {ccc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if relative_rmse is not None and relative_rmse_found > relative_rmse:
        msg = f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} % <= {relative_rmse * 100:.1f} %"
        raise Exception(msg)  # noqa: TRY002


@pytest.mark.parametrize("seq_len", [686, 704])
@pytest.mark.parametrize("n_blocks", [64])
def test_pairformer(seq_len, n_blocks, ttnn_only=True):
    pairformer = PairformerModule(
        n_blocks=n_blocks,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
        transform_s=True,
    )
    
    pairformer_torch = PairformerModuleTorch(
        token_s=384, token_z=128, num_blocks=n_blocks, v2=True
    ).eval()

    logger.info("Loading Models")
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = mask[:, :, None] * mask[:, None, :]

    if ttnn_only:
        logger.info(f"TTNN-ONLY mode for Pairformer (seq_len={seq_len}, n_blocks={n_blocks})")
        start_time = time.time()
        s_tt, z_tt = pairformer(s, z, mask, pair_mask)
        ttnn_time = time.time() - start_time
        logger.info(f"TTNN Pairformer (seq_len={seq_len}, n_blocks={n_blocks}): {ttnn_time:.4f}s")
        return

    # Always run Torch first, cache results for TTNN comparison
    logger.info("Torch Forward Pass")
    torch_cache = {}
    start_time = time.time()
    s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    torch_time = time.time() - start_time
    logger.info(f"Torch Pairformer (seq_len={seq_len}, n_blocks={n_blocks}): {torch_time:.4f}s")
    torch_cache["s_torch"] = s_torch
    torch_cache["z_torch"] = z_torch

    logger.info("TTNN Forward Pass")
    start_time = time.time()
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    ttnn_time = time.time() - start_time
    logger.info(f"Pairformer (seq_len={seq_len}, n_blocks={n_blocks}): Torch: {torch_time:.4f}s | TTNN: {ttnn_time:.4f}s | Speedup: {torch_time/ttnn_time:.2f}x")
    assert_quality(s_tt, torch_cache["s_torch"], pcc=0.98)
    assert_quality(z_tt, torch_cache["z_torch"], pcc=0.98)

@pytest.mark.parametrize("seq_len", [686, 704])
@pytest.mark.parametrize("n_layers", [24])
def test_token_transformer(n_layers, seq_len, ttnn_only=True):
    token_transformer = DiffusionTransformerModule( 
        n_layers=n_layers,
        dim=768,
        n_heads=16,
        atom_level=False
    )
    token_transformer_torch = DiffusionTransformerTorch(
        depth=n_layers, heads=16, dim=768, dim_single_cond=768
    ).eval()
    token_transformer_state_dict = filter_dict(
        state_dict, "structure_module.score_model.token_transformer"
    )
    logger.info("Loading Model")
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)
    a = 3 + 5 * torch.randn(1, seq_len, 768)
    s = -2 + 42 * torch.randn(1, seq_len, 768)
    z = 10 * torch.randn(1, seq_len, seq_len, n_layers * 16)
    mask = torch.ones(1, seq_len)

    if ttnn_only:
        logger.info(f"TTNN-ONLY mode for TokenTransformer (seq_len={seq_len}, n_layers={n_layers})")
        start_time = time.time()
        a_tt = token_transformer(a, s, z, mask)
        ttnn_time = time.time() - start_time
        logger.info(f"TTNN TokenTransformer (seq_len={seq_len}, n_layers={n_layers}): {ttnn_time:.4f}s")
        return

    # Always run Torch first, cache result for TTNN comparison
    logger.info("Torch Forward Pass")
    torch_cache = {}
    start_time = time.time()
    a_torch = token_transformer_torch(a, s, z, mask)
    torch_time = time.time() - start_time
    torch_cache["a_torch"] = a_torch

    logger.info("TTNN Forward Pass")
    start_time = time.time()
    a_tt = token_transformer(a, s, z, mask)
    ttnn_time = time.time() - start_time
    logger.info(f"TokenTransformer (seq_len={seq_len}, n_layers={n_layers}): Torch: {torch_time:.4f}s | TTNN: {ttnn_time:.4f}s | Speedup: {torch_time/ttnn_time:.2f}x")
    assert_quality(a_tt, torch_cache["a_torch"], pcc=0.99)

@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_layers", [3])
def test_atom_transformer(n_layers, n_heads, ttnn_only=True):
    atom_transformer = DiffusionTransformerModule(
        n_layers=n_layers,
        dim=128,
        n_heads=n_heads,
        atom_level=True
    )
    atom_transformer_torch = DiffusionTransformerTorch(
        depth=n_layers, heads=n_heads, dim=128, dim_single_cond=128
    ).eval()
    atom_transformer_state_dict = filter_dict(
        state_dict,
        "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer",
    )
    logger.info("Loading Model")
    atom_transformer.load_state_dict(
        atom_transformer_state_dict,
        strict=False,
    )
    atom_transformer_torch.load_state_dict(atom_transformer_state_dict, strict=False)
    B, W, H, K = 1, 32, 128, 29
    a = torch.randn(K, W, H)
    s = torch.randn(K, W, H)
    bias = torch.randn(K, W, H, n_layers * n_heads)
    mask = torch.ones(K, W)
    keys_indexing = get_indexing_matrix(K, W, H, "cpu")
    to_keys = partial(single_to_keys, indexing_matrix=keys_indexing, W=W, H=H)
    to_keys_new = lambda x: to_keys(x.view(B, K * W, -1)).view(K, H, -1)

    if ttnn_only:
        logger.info(f"TTNN-ONLY mode for AtomTransformer (n_layers={n_layers}, n_heads={n_heads}, K={K}, W={W}, H={H})")
        start_time = time.time()
        a_tt = atom_transformer(a, s, bias, mask, keys_indexing)
        ttnn_time = time.time() - start_time
        logger.info(f"TTNN AtomTransformer (n_layers={n_layers}, n_heads={n_heads}, K={K}, W={W}, H={H}): {ttnn_time:.4f}s")
        return

    # Always run Torch first, cache result for TTNN comparison
    logger.info("Torch Forward Pass")
    torch_cache = {}
    start_time = time.time()
    a_torch = atom_transformer_torch(a, s, bias, mask, to_keys_new)
    torch_time = time.time() - start_time
    torch_cache["a_torch"] = a_torch

    logger.info("TTNN Forward Pass")
    start_time = time.time()
    a_tt = atom_transformer(a, s, bias, mask, keys_indexing)
    ttnn_time = time.time() - start_time
    logger.info(f"AtomTransformer (n_layers={n_layers}, n_heads={n_heads}, K={K}, W={W}, H={H}): Torch: {torch_time:.4f}s | TTNN: {ttnn_time:.4f}s | Speedup: {torch_time/ttnn_time:.2f}x")
    assert_quality(a_tt, torch_cache["a_torch"], pcc=0.99)


def filter_msa_state_dict_for_blocks(state_dict, n_blocks):
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith("layers."):
            block_idx = int(k.split(".")[1])
            if block_idx < n_blocks:
                filtered[k] = v
        else:
            filtered[k] = v
    return filtered

@pytest.mark.parametrize("seq_len", [686])
@pytest.mark.parametrize("n_sequences", [1])
@pytest.mark.parametrize("n_blocks", [4])
def test_msa(seq_len, n_sequences, n_blocks, ttnn_only=True):
    from boltz.model.modules.tenstorrent import timing_stats, shape_timing_stats

    msa = MSAModule(
        n_blocks=n_blocks,
        avg_head_dim=32,
        avg_n_heads=8,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
    )
    msa_torch = MSAModuleTorch(msa_s=64, token_z=128, token_s=384, msa_blocks=n_blocks, msa_dropout=0, z_dropout=0).eval()
    msa_state_dict = filter_dict(state_dict, "msa_module")
    msa_state_dict = filter_msa_state_dict_for_blocks(msa_state_dict, n_blocks)
    logger.info("Loading Model")
    msa.load_state_dict(msa_state_dict, strict=False)
    msa_torch.load_state_dict(msa_state_dict, strict=False)

    # Create inputs once to reuse
    z = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 384)
    feats = {"msa": torch.randint(33, (1, n_sequences, seq_len)),
             "has_deletion": torch.zeros((1, n_sequences, seq_len), dtype=torch.bool),
             "deletion_value": torch.zeros((1, n_sequences, seq_len)),
             "msa_paired": torch.zeros((1, n_sequences, seq_len)),
             "msa_mask": torch.ones((1, n_sequences, seq_len)),
             "token_pad_mask": torch.ones((1, seq_len)),
    }

    if ttnn_only:
        logger.info(f"TTNN-ONLY mode for MSA (seq_len={seq_len}, n_sequences={n_sequences}, n_blocks={n_blocks})")
        
        # Warmup run
        logger.info("Warmup run...")
        z_tt = msa(z, emb, feats)
        
        # Clear timing stats
        timing_stats.clear()
        shape_timing_stats.clear()
        
        # Multiple timed runs
        n_runs = 4
        logger.info(f"Starting {n_runs} timed runs...")
        for i in range(n_runs):
            start_time = time.time()
            z_tt = msa(z, emb, feats)
            run_time = time.time() - start_time
            logger.info(f"Run {i+1}/{n_runs}: {run_time:.4f}s")
        
        return

    # Always run Torch first, cache result for TTNN comparison
    logger.info("Torch Forward Pass")
    torch_cache = {}
    start_time = time.time()
    z_torch = msa_torch(z, emb, feats)
    torch_time = time.time() - start_time
    torch_cache["z_torch"] = z_torch

    logger.info("TTNN Forward Pass")
    start_time = time.time()
    z_tt = msa(z, emb, feats)
    ttnn_time = time.time() - start_time
    logger.info(f"MSA (seq_len={seq_len}, n_sequences={n_sequences}, n_blocks={n_blocks}): Torch: {torch_time:.4f}s | TTNN: {ttnn_time:.4f}s | Speedup: {torch_time/ttnn_time:.2f}x")
    assert_quality(z_tt, torch_cache["z_torch"], pcc=0.99)