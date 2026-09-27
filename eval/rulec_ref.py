#!/usr/bin/env python3
"""Rule-C reference labels + Step-4 parity for the Nemotron split (plans/2026-09-26-nemotron-split-naming.md).

The evaluation data (meeting audio, voiceprints, blind-test keys) is private and lives OUTSIDE the
repo, in the regression set (default ~/.meetscribe/eval/nemotron-split). This script only reads it.

Reference = the rule that won the blind test: for every Nemotron-split window B (from split_probe's
B.out), a window >= 1.5 s takes B's own speaker; a shorter one takes the speaker the whole Whisper
chunk got on path A (A.out, chunk index = B key // 1000).

    rulec_ref.py ref    <meeting-id> [--eval DIR]                    -> reference windows as JSON
    rulec_ref.py parity <meeting-id> <transcript.json> [--eval DIR]  -> time-weighted agreement

`parity` compares the far-end names in a meetscribe JSON export against the reference, weighted by
overlap time, and exits 1 below the 99 % gate.
"""

import argparse
import json
import os
import sys

MIN_WINDOW_SECS = 1.5  # spk::MIN_WINDOW_MS
GATE = 0.99


def load(path):
    with open(path) as f:
        return json.load(f)


def reference(eval_dir, meeting):
    split = os.path.join(eval_dir, "split")
    names = {int(k): v for k, v in load(os.path.join(split, "names.json")).items()}
    name = lambda sid: names.get(sid, f"speaker#{sid}") if sid else "Others"
    a = load(os.path.join(split, f"{meeting}.A.out.json"))["windows"]
    b = load(os.path.join(split, f"{meeting}.B.out.json"))["windows"]
    a_name = {w["key"]: name(w["speaker_id"]) for w in a}
    out = []
    for w in b:
        long_enough = w["t_end"] - w["t_start"] >= MIN_WINDOW_SECS
        who = name(w["speaker_id"]) if long_enough else a_name[w["key"] // 1000]
        out.append({"t_start": w["t_start"], "t_end": w["t_end"], "name": who})
    return out


def exported_far_end(path):
    segs = load(path)
    segs = segs["segments"] if isinstance(segs, dict) else segs
    return [
        (s["t_start"], s["t_end"], s.get("speaker_name") or "Others")
        for s in segs
        if s["speaker"] != "you"
    ]


def parity(ref, exported):
    agree = total = 0.0
    confusion = {}
    for w in ref:
        for a, b, who in exported:
            o = min(b, w["t_end"]) - max(a, w["t_start"])
            if o <= 0:
                continue
            total += o
            if who == w["name"]:
                agree += o
            else:
                key = (w["name"], who)
                confusion[key] = confusion.get(key, 0.0) + o
    ref_secs = sum(w["t_end"] - w["t_start"] for w in ref)
    return agree, total, ref_secs, confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ref", "parity"])
    p.add_argument("meeting")
    p.add_argument("export", nargs="?")
    p.add_argument("--eval", default=os.path.expanduser("~/.meetscribe/eval/nemotron-split"))
    args = p.parse_args()

    ref = reference(args.eval, args.meeting)
    if args.cmd == "ref":
        json.dump(ref, sys.stdout, ensure_ascii=False, indent=1)
        return 0
    if not args.export:
        p.error("parity needs the exported transcript.json")
    agree, total, ref_secs, confusion = parity(ref, exported_far_end(args.export))
    share = agree / total if total else 0.0
    print(f"{args.meeting}: agreement {share:.2%} over {total:.0f}s compared "
          f"({ref_secs - total:.0f}s of reference time not covered by any exported far-end segment)")
    for (want, got), secs in sorted(confusion.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  ref {want} -> export {got}: {secs:.1f}s")
    ok = share >= GATE
    print("PASS" if ok else f"FAIL (gate {GATE:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
