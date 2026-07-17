"""
VLM Token Contract — save/restore special token embeddings across processes.
===================================================================
Ensures P1 and P2 see identical VLM hidden states by preserving the
exact token IDs, tokenizer vocab size, and embedding weights for all
custom tokens added to the base VLM.

Usage:
  # P1: save alongside backbone checkpoint
  contract = build_vlm_contract(vla)
  torch.save({"p1_state_dict": ..., "vlm_contract": contract}, path)

  # P2: restore before any VLM forward
  contract = torch.load(path)["vlm_contract"]
  restore_vlm_contract(vla, contract)  # raises if checks fail
"""

import hashlib
from typing import Dict, List, Tuple

import torch

# Tokens that must be in the contract
_REQUIRED_SPECIAL_TOKENS = [
    "<|thinking|>",
    "<|start_of_thinking|>",
    "<|end_of_thinking|>",
    "<img_next>",
]

CONTRACT_VERSION = 1


def build_vlm_contract(vla) -> dict:
    """
    Build a VLM token contract from the current VLA instance.

    Returns a dict that can be serialized alongside the backbone checkpoint.
    """
    tokenizer = vla.qwen_vl_interface.tokenizer
    embed = vla.qwen_vl_interface.model.get_input_embeddings()
    embed_weight = embed.weight.data

    special_tokens = {}
    for token_name in _REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None:
            raise ValueError(f"Special token '{token_name}' not found in tokenizer")
        special_tokens[token_name] = {
            "token_id": int(token_id),
            "embedding": embed_weight[token_id].cpu().clone(),
        }

    contract = {
        "contract_version": CONTRACT_VERSION,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "embedding_dim": int(embed_weight.shape[1]),
        "special_tokens": special_tokens,
        # Hash of all special token embeddings combined
        "embedding_hash": _hash_embeddings(special_tokens),
    }
    return contract


def restore_vlm_contract(vla, contract: dict) -> bool:
    """
    Restore VLM special token embeddings from a contract.

    Verifies: contract version, token IDs match, embedding shapes match.
    Returns True if successfully restored, raises on mismatch.
    """
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError(
            f"VLM contract version mismatch: saved={contract.get('contract_version')}, "
            f"expected={CONTRACT_VERSION}")

    tokenizer = vla.qwen_vl_interface.tokenizer
    embed = vla.qwen_vl_interface.model.get_input_embeddings()
    embed_weight = embed.weight.data

    # Check embedding dimension
    saved_dim = contract["embedding_dim"]
    current_dim = int(embed_weight.shape[1])
    if saved_dim != current_dim:
        raise RuntimeError(
            f"VLM embedding dim mismatch: saved={saved_dim}, current={current_dim}")

    special_tokens = contract["special_tokens"]
    for token_name in _REQUIRED_SPECIAL_TOKENS:
        if token_name not in special_tokens:
            raise RuntimeError(f"Missing special token '{token_name}' in contract")

        token_info = special_tokens[token_name]
        saved_id = token_info["token_id"]
        saved_emb = token_info["embedding"]

        # Verify token ID matches
        current_id = tokenizer.convert_tokens_to_ids(token_name)
        if isinstance(current_id, list):
            current_id = current_id[0] if current_id else None
        if current_id != saved_id:
            raise RuntimeError(
                f"Token ID mismatch for '{token_name}': saved={saved_id}, current={current_id}")

        # Verify shape
        if saved_emb.shape[0] != embed_weight.shape[1]:
            raise RuntimeError(
                f"Embedding dim mismatch for '{token_name}': "
                f"saved={saved_emb.shape[0]}, current={embed_weight.shape[1]}")

    # Restore embeddings for all special tokens
    for token_name in _REQUIRED_SPECIAL_TOKENS:
        token_info = special_tokens[token_name]
        token_id = token_info["token_id"]
        embed_weight[token_id].copy_(token_info["embedding"].to(embed_weight.device))

    # Verify hash after restore
    current_hash = _hash_embeddings(special_tokens, embed_weight)
    saved_hash = contract["embedding_hash"]
    if current_hash != saved_hash:
        raise RuntimeError(
            f"Embedding hash mismatch after restore: saved={saved_hash}, current={current_hash}")

    return True


def _hash_embeddings(special_tokens: dict, embed_weight=None) -> str:
    """Compute SHA256 hash of special token embeddings."""
    h = hashlib.sha256()
    for name in sorted(special_tokens.keys()):
        h.update(name.encode())
        info = special_tokens[name]
        h.update(str(info["token_id"]).encode())
        emb = info["embedding"]
        if embed_weight is not None:
            # Use current weight to verify after restore
            emb = embed_weight[info["token_id"]]
        h.update(emb.cpu().numpy().tobytes())
    return h.hexdigest()[:16]
