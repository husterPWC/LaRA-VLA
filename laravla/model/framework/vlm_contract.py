"""
VLM Token Contract — save/restore full VLM input embeddings across processes.
=====================================================================
Saves the ENTIRE input embedding matrix (not just special tokens) to
ensure P1 and P2 see identical VLM hidden states. `resize_token_embeddings`
extends the matrix with random rows; saving the full matrix captures all
of them.

Usage:
  # P1: save with backbone checkpoint
  contract = build_vlm_contract(vla)
  torch.save({"p1_state_dict": ..., "vlm_contract": contract}, path)

  # P2: restore before any VLM forward
  contract = torch.load(path)["vlm_contract"]
  restore_vlm_contract(vla, contract)
"""

import hashlib
import torch

CONTRACT_VERSION = 2


def build_vlm_contract(vla) -> dict:
    """Save the full VLM input embedding matrix."""
    embed = vla.qwen_vl_interface.model.get_input_embeddings()
    embed_weight = embed.weight.data.cpu().clone()  # keep native dtype (bf16)

    # Verify: special tokens must exist
    tokenizer = vla.qwen_vl_interface.processor.tokenizer
    special_ids = {}
    for name in ["<|thinking|>", "<|start_of_thinking|>", "<|end_of_thinking|>", "<img_next>"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, list):
            tid = tid[0] if tid else None
        if tid is None:
            raise ValueError(f"Special token '{name}' not found")
        special_ids[name] = int(tid)

    h = hashlib.sha256()
    h.update(embed_weight.float().numpy().tobytes())
    embed_hash = h.hexdigest()[:16]

    return {
        "contract_version": CONTRACT_VERSION,
        "embedding_weight": embed_weight,           # [vocab_size, embed_dim] float32
        "special_token_ids": special_ids,            # {name: token_id}
        "embedding_hash": embed_hash,
    }


def restore_vlm_contract(vla, contract: dict) -> bool:
    """Restore full embedding matrix. Returns True on success."""
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError(
            f"Contract version mismatch: {contract.get('contract_version')} != {CONTRACT_VERSION}")

    saved_weight = contract["embedding_weight"]
    embed = vla.qwen_vl_interface.model.get_input_embeddings()
    current_weight = embed.weight.data

    if saved_weight.shape != current_weight.shape:
        raise RuntimeError(
            f"Embedding shape mismatch: saved={saved_weight.shape} current={current_weight.shape}")

    # Copy entire embedding matrix
    current_weight.copy_(saved_weight.to(current_weight.device, current_weight.dtype))

    # Verify hash
    h = hashlib.sha256()
    h.update(current_weight.cpu().float().numpy().tobytes())
    current_hash = h.hexdigest()[:16]
    if current_hash != contract["embedding_hash"]:
        raise RuntimeError(
            f"Hash mismatch after restore: saved={contract['embedding_hash']} current={current_hash}")

    return True
