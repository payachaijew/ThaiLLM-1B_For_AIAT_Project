"""Helpers for the custom Thai evaluation tasks.

M3Exam stores the answer as a STRING containing the option number ("4"), while the options
themselves are strings prefixed with that number ("4. ..."). lm-eval needs an integer index
into doc_to_choice, so the mapping is done here rather than in a fragile Jinja expression.
Rows whose answer does not resolve to a valid option are dropped and counted, because a
silently mis-indexed answer would score the model against the wrong string.
"""
from __future__ import annotations
import re


def process_m3exam(dataset):
    dropped = {"n": 0}

    def _map(doc):
        opts = list(doc.get("options") or [])
        ans = str(doc.get("answer_text", "")).strip()
        target = -1
        if ans.isdigit():
            for i, o in enumerate(opts):
                m = re.match(r"\s*(\d+)", str(o))
                if m and m.group(1) == ans:
                    target = i
                    break
            if target < 0 and 1 <= int(ans) <= len(opts):
                target = int(ans) - 1          # fall back to positional numbering
        doc["target_index"] = target
        return doc

    ds = dataset.map(_map)
    before = len(ds)
    ds = ds.filter(lambda d: d["target_index"] >= 0 and len(d.get("options") or []) >= 2)
    dropped["n"] = before - len(ds)
    if dropped["n"]:
        print(f"[m3exam_th] dropped {dropped['n']} of {before} rows with unresolvable answers")
    return ds


def _drop_incomplete(dataset, keys):
    """Drop rows where any option is blank or the answer is not one of the options.

    ThaiExam ships some rows with empty option fields. Left in place the model would be
    scored on choosing between blank strings, which is noise presented as accuracy.
    """
    before = len(dataset)
    ds = dataset.filter(
        lambda d: all(str(d.get(k, "")).strip() for k in keys)
        and str(d.get("answer", "")).strip() in keys
    )
    if len(ds) != before:
        print(f"[thaiexam] dropped {before - len(ds)} of {before} rows with blank options "
              f"or an out-of-range answer")
    return ds


def process_thaiexam_5(dataset):
    return _drop_incomplete(dataset, ["a", "b", "c", "d", "e"])


def process_thaiexam_4(dataset):
    return _drop_incomplete(dataset, ["a", "b", "c", "d"])
