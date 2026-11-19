"""ESM-2 zero-shot scoring of the cytosine deaminase variants.

Position bookkeeping
--------------------
Three coordinate systems have to line up, and getting this wrong is silent:

    canonical position   (what the lab says: 126, 314)
    index in our strings (canonical - SEQ_OFFSET - 1, zero-based)
    ESM token index      (string index + 1, because of the BOS token)

`token_index()` does the whole conversion. `check_position()` asserts the
residue found there is the one expected, so a mistake fails loudly.

Scoring schemes
---------------
Two schemes, and the difference between them is the whole point of this
notebook:

`wt_marginals`
    Each mutated position is masked in the WILD-TYPE sequence and scored
    independently; the double mutant is the sum of its singles. This is
    additive BY CONSTRUCTION and cannot express coupling.

`mutant_marginals`
    Each mutated position is masked in a sequence that already carries the
    OTHER mutation. The model sees the partner mutation in context, so this
    scheme CAN express coupling. Whether it does is an empirical question.

A third, `delta_pll`, scores the whole sequence (every position masked in turn)
and is fully non-additive, at the cost of one forward pass per residue.
"""

from typing import Dict, List, Sequence, Tuple

from .sequences import SEQ_OFFSET

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
TOY_MODEL = "facebook/esm2_t12_35M_UR50D"   # for a smoke test on a laptop


def string_index(canonical_pos: int) -> int:
    """Zero-based index into the sequences as provided by the lab."""
    return canonical_pos - SEQ_OFFSET - 1


def token_index(canonical_pos: int) -> int:
    """Index into the ESM token sequence, accounting for the BOS token."""
    return string_index(canonical_pos) + 1


def check_position(seq: str, canonical_pos: int, expected_aa: str) -> None:
    found = seq[string_index(canonical_pos)]
    if found != expected_aa:
        raise AssertionError(
            f"expected {expected_aa} at canonical {canonical_pos}, found {found}")


def apply_mutations(seq: str, muts: Sequence[Tuple[int, str, str]]) -> str:
    out = list(seq)
    for pos, wt_aa, mut_aa in muts:
        check_position(seq, pos, wt_aa)
        out[string_index(pos)] = mut_aa
    return "".join(out)


def load_model(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval().to(device)
    return tok, model, device


def _masked_logprobs(tok, model, device, seq: str, tok_idx: int):
    """log-softmax over the vocabulary at one masked token position."""
    import torch

    enc = tok(seq, return_tensors="pt").to(device)
    ids = enc["input_ids"].clone()
    ids[0, tok_idx] = tok.mask_token_id
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=enc["attention_mask"]).logits
    return torch.log_softmax(logits[0, tok_idx], dim=-1)


def llr_at(tok, model, device, seq: str, canonical_pos: int,
           wt_aa: str, mut_aa: str) -> float:
    """log P(mut) - log P(wt) at one position, with that position masked."""
    lp = _masked_logprobs(tok, model, device, seq, token_index(canonical_pos))
    return float(lp[tok.convert_tokens_to_ids(mut_aa)]
                 - lp[tok.convert_tokens_to_ids(wt_aa)])


def score_variant(tok, model, device, wt_seq: str,
                  muts: Sequence[Tuple[int, str, str]],
                  scheme: str = "wt_marginals") -> float:
    """Total score of a variant under one scheme. WT scores exactly 0."""
    if not muts:
        return 0.0
    total = 0.0
    for pos, wt_aa, mut_aa in muts:
        if scheme == "wt_marginals":
            context = wt_seq
        elif scheme == "mutant_marginals":
            others = [m for m in muts if m[0] != pos]
            context = apply_mutations(wt_seq, others)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        total += llr_at(tok, model, device, context, pos, wt_aa, mut_aa)
    return total


def pseudo_log_likelihood(tok, model, device, seq: str, batch_size: int = 16) -> float:
    """Sum over all positions of log P(aa_i | seq with position i masked).

    One forward pass per residue. For a 423-residue protein that is 423 passes;
    batched on a GPU it is seconds, on CPU it is minutes.
    """
    import torch

    enc = tok(seq, return_tensors="pt")
    ids = enc["input_ids"][0]
    positions = list(range(1, len(ids) - 1))  # skip BOS and EOS
    total = 0.0

    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        batch = ids.repeat(len(chunk), 1)
        for row, p in enumerate(chunk):
            batch[row, p] = tok.mask_token_id
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(input_ids=batch).logits
        lp = torch.log_softmax(logits, dim=-1)
        for row, p in enumerate(chunk):
            total += float(lp[row, p, ids[p]])
    return total
