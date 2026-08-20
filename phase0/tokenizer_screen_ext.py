#!/usr/bin/env python3
"""Phase 0 / Step A -- extend the frozen tokenizer screen to the live candidates.

Reuses the EXACT frozen sample set from base_selection/tokenizer_screen.json
(verified per-sample by sha256) so every number is directly comparable with the
existing BASE-SCREEN-2026-08-18-TOKENIZER artifact.

Adds the two candidates that screen predates:
  * Qwen/Qwen3-1.7B-Base   -- the proposed main base
  * google/gemma-4-E2B     -- the challenger, Apache-2.0 since 2026-04
  * Qwen/Qwen3-4B-Base     -- in-family scale-up option

Re-measures Qwen/Qwen3-0.6B-Base as a METHODOLOGY CONTROL: its numbers must
reproduce the original screen exactly, otherwise this artifact is not comparable.

scientific_evidence_allowed = false. This is a deterministic tokenizer
measurement. No model weights are loaded. It measures COST, not quality.
"""
from __future__ import annotations
import json, hashlib, sys, datetime, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCREEN = ROOT.parent / "base_selection" / "tokenizer_screen.json"
OUT = ROOT / "tokenizer_screen_ext.json"

NEW_CANDIDATES = [
    ("Qwen/Qwen3-1.7B-Base", "proposed_main_base"),
    ("google/gemma-4-E2B",   "challenger_apache_since_2026_04"),
    ("Qwen/Qwen3-4B-Base",   "in_family_scale_up_option"),
    ("Qwen/Qwen3-0.6B-Base", "METHODOLOGY_CONTROL_must_reproduce_original_screen"),
]

THAI = lambda ch: "\u0e00" <= ch <= "\u0e7f"


def _byte_decoder():
    """Reverse of the GPT-2 byte-to-unicode map used by byte-level BPE vocabs."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BD = _byte_decoder()


def surface(piece: str, byte_level: bool) -> str:
    """Return the human-readable text a vocab piece stands for."""
    if not byte_level:
        return piece.replace("\u2581", " ")
    try:
        return bytes(_BD[c] for c in piece).decode("utf-8", errors="ignore")
    except KeyError:
        return piece


def load_samples():
    d = json.loads(SCREEN.read_text())
    samples = d["sample_set"]["samples"]
    bad = [s["sample_id"] for s in samples
           if hashlib.sha256(s["text"].encode()).hexdigest() != s["sha256"]]
    if bad:
        sys.exit(f"FATAL: frozen sample integrity failure: {bad}")
    return d, samples


def resolve(repo):
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(repo)
        return info.sha, getattr(info, "gated", None)
    except Exception as e:
        return None, f"resolve_failed: {type(e).__name__}: {e}"


def get_tokenizer(repo, revision):
    """AutoTokenizer first; fall back to raw tokenizer.json for architectures
    this transformers version does not yet know (e.g. gemma4)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
        return tok, "transformers.AutoTokenizer", None
    except Exception as e_auto:
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            p = hf_hub_download(repo, "tokenizer.json", revision=revision)
            return Tokenizer.from_file(p), "tokenizers.Tokenizer(raw tokenizer.json)", \
                   f"AutoTokenizer unavailable: {type(e_auto).__name__}: {str(e_auto)[:200]}"
        except Exception as e_raw:
            raise RuntimeError(f"both loaders failed: auto={e_auto!r} raw={e_raw!r}")


def encode(tok, text):
    if hasattr(tok, "encode") and hasattr(tok, "get_vocab") and not hasattr(tok, "convert_ids_to_tokens"):
        return tok.encode(text, add_special_tokens=False).ids   # tokenizers.Tokenizer
    try:
        return tok.encode(text, add_special_tokens=False)  # transformers
    except TypeError:
        return tok.encode(text, add_special_tokens=False).ids


def decode(tok, ids):
    try:
        return tok.decode(ids, skip_special_tokens=False)
    except TypeError:
        return tok.decode(ids)


def vocab_of(tok):
    try:
        return tok.get_vocab()
    except Exception:
        return {}


def measure(repo, role, samples):
    rev, gated = resolve(repo)
    tok, loader, warn = get_tokenizer(repo, rev)
    vocab = vocab_of(tok)

    byte_level = any("\u0120" in p for p in list(vocab)[:5000])
    surfaces_map = {p: surface(p, byte_level) for p in vocab}
    thai_pieces = [s for s in surfaces_map.values() if any(THAI(c) for c in s)]
    multichar = [s for s in thai_pieces if sum(1 for c in s if THAI(c)) > 1]
    longest = max(thai_pieces, key=lambda s: sum(1 for c in s if THAI(c)), default=None)

    per_doc, groups, roundtrip_fail, repl, unk = {}, {}, [], 0, 0
    for s in samples:
        ids = encode(tok, s["text"])
        n = len(ids)
        b, c = len(s["text"].encode()), len(s["text"])
        per_doc[s["sample_id"]] = {
            "tokens": n,
            "utf8_bytes": b,
            "chars": c,
            "bytes_per_token": round(b / n, 4) if n else None,
            "chars_per_token": round(c / n, 4) if n else None,
        }
        if decode(tok, ids) != s["text"]:
            roundtrip_fail.append(s["sample_id"])
        surfaces = "".join(decode(tok, [i]) for i in ids)
        repl += surfaces.count("�")
        groups.setdefault(s["language"], []).append((b, c, n))

    agg = {}
    for lang, rows in groups.items():
        B, C, N = (sum(r[i] for r in rows) for i in range(3))
        agg[lang] = {
            "documents": len(rows),
            "total_tokens": N,
            "bytes_per_token": round(B / N, 4),
            "chars_per_token": round(C / N, 4),
            "tokens_per_document": round(N / len(rows), 2),
        }

    th_par, en_par = per_doc["TH-PAR-01"]["tokens"], per_doc["EN-PAR-01"]["tokens"]
    return {
        "canonical_repo": repo,
        "role": role,
        "revision": rev,
        "gated": gated,
        "loader": loader,
        "loader_warning": warn,
        "vocab_size": len(vocab) if vocab else None,
        "byte_level_bpe": byte_level,
        "vocab_pieces_containing_thai": len(thai_pieces),
        "vocab_pieces_multichar_thai": len(multichar),
        "pct_vocab_containing_thai": round(100 * len(thai_pieces) / len(vocab), 3) if vocab else None,
        "longest_thai_piece": longest,
        "longest_thai_piece_thai_chars": sum(1 for c in longest if THAI(c)) if longest else 0,
        "aggregate_by_language": agg,
        "thai_vs_english_token_ratio_parallel_doc": round(th_par / en_par, 4),
        "total_replacement_char_tokens": repl,
        "total_unk_tokens": unk,
        "roundtrip_lossless_all_docs": not roundtrip_fail,
        "roundtrip_failures": roundtrip_fail,
        "per_document": per_doc,
    }


def main():
    orig, samples = load_samples()
    results, errors = {}, {}
    for repo, role in NEW_CANDIDATES:
        print(f"[*] {repo} ...", flush=True)
        try:
            results[repo] = measure(repo, role, samples)
            r = results[repo]
            print(f"    thai chars/token={r['aggregate_by_language']['th']['chars_per_token']} "
                  f"th/en ratio={r['thai_vs_english_token_ratio_parallel_doc']} vocab={r['vocab_size']}")
        except Exception as e:
            errors[repo] = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {errors[repo]}")

    # methodology control
    ctrl_repo = "Qwen/Qwen3-0.6B-Base"
    control = {"status": "not_run"}
    if ctrl_repo in results:
        prev = orig["models"].get(ctrl_repo, {})
        now = results[ctrl_repo]
        checks = {
            "thai_chars_per_token": (prev.get("aggregate_by_language", {}).get("th", {}).get("chars_per_token"),
                                     now["aggregate_by_language"]["th"]["chars_per_token"]),
            "thai_vs_english_token_ratio_parallel_doc": (prev.get("thai_vs_english_token_ratio_parallel_doc"),
                                                         now["thai_vs_english_token_ratio_parallel_doc"]),
            "vocab_pieces_containing_thai": (prev.get("vocab_pieces_containing_thai"),
                                             now["vocab_pieces_containing_thai"]),
        }
        control = {
            "status": "pass" if all(a == b for a, b in checks.values()) else "MISMATCH",
            "comparisons": {k: {"original_screen": a, "this_run": b, "match": a == b}
                            for k, (a, b) in checks.items()},
            "note": "If MISMATCH, this artifact is NOT comparable with the original screen.",
        }

    # cost projection anchored on SEA-PILE-v2 th
    anchor_bpt = None
    for m in orig["models"].values():
        if m["canonical_repo"] == "google/gemma-3-1b-pt":
            anchor_bpt = m["aggregate_by_language"]["th"]["bytes_per_token"]
    proj = {}
    if anchor_bpt:
        for repo, r in results.items():
            bpt = r["aggregate_by_language"]["th"]["bytes_per_token"]
            mult = anchor_bpt / bpt
            proj[repo] = {
                "thai_bytes_per_token": bpt,
                "multiplier_vs_gemma3_anchor": round(mult, 4),
                "sea_pile_v2_th_tokens_estimate": round(6.5e9 * mult / 1e9, 2),
                "unit": "billions of tokens for the same Thai corpus bytes",
            }

    out = {
        "screen_id": "PHASE0-A-TOKENIZER-EXT-2026-08-18",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "phase0_tokenizer_only",
        "scientific_evidence_allowed": False,
        "evidence_class": "deterministic_tokenizer_measurement",
        "extends": orig["screen_id"],
        "sample_set_reused": {
            "sample_set_id": orig["sample_set"]["sample_set_id"],
            "n_samples": len(samples),
            "per_sample_sha256_verified": True,
            "note": "Identical frozen texts as the original screen; verified per sample by sha256.",
        },
        "metrics_defined": orig["metrics_defined"],
        "methodology_control": control,
        "models": results,
        "errors": errors,
        "corpus_cost_projection": {
            "anchor": "SEA-PILE-v2 subset th, 6.5B tokens under the Gemma 3 tokenizer per the dataset card",
            "interpretation": "Same corpus bytes, different token counts. Higher multiplier = same Thai text costs more compute.",
            "per_model": proj,
        },
        "limitations": [
            "14 documents is a screening sample, not a corpus-scale fertility estimate. The RANKING is what this supports.",
            "Fertility is a COST metric. It does not predict how well a model learns Thai.",
            "No model weights were loaded. No BPB, perplexity or downstream number appears here.",
            "Thai token estimates for SEA-PILE-v2 are projections from a 14-document sample, not a recount of the corpus.",
        ],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[+] wrote {OUT}")
    print(f"[+] methodology control: {control.get('status')}")


if __name__ == "__main__":
    main()
