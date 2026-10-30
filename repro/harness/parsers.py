"""Parse optimizer outputs into (time series, events, final circuit counts).

Every parser returns a dict with
  series:   list of (t_s, total, cx) change-points, best-so-far as reported by
            the tool (includes points after the budget; callers truncate)
  kevents:  list of (k, t_s)  [QALM only]
  greedy:   list of (pass_name, t_s, cost_before, cost_after)  [QALM/Quartz]
  final:    gate counts of the final circuit written by the tool, or None
  extra:    tool-specific facts
"""
import gzip
import json
import os
import re

from qasm_count import count_text

_PROGRESS = re.compile(
    r"^\[(?P<name>.+?)\] Best cost: (?P<cost>[0-9.]+)\tCX: (?P<cx>\d+)\t"
    r"candidate number: \d+\tafter (?P<t>[0-9.]+) seconds\.$")
_KSTART = re.compile(r"^\[(?P<name>.+?)\] \[QALM_K_START\] k=(?P<k>\d+) at (?P<t>[0-9.]+) seconds\.$")
_GREEDY = re.compile(
    r"^(?P<pass>greedy_optimize_with_\w+)\(\): cost optimized from (?P<a>[0-9.]+) to (?P<b>[0-9.]+) "
    r"after (?P<t>[0-9.]+) seconds\.$")
_POST_GREEDY_T = re.compile(r"^Post-greedy time: (?P<t>[0-9.]+) seconds")
_GREEDY_TIMEOUT = re.compile(r"^greedy_optimize\(\): timeout hit after (?P<t>[0-9.]+)")


def open_text(path):
    """Open a log that may have been gzipped after collection."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", errors="replace")
    return open(path, errors="replace")


def _append(series, t, total, cx):
    if not series or (series[-1][1], series[-1][2]) != (total, cx):
        series.append((t, total, cx))


def parse_native(stdout_path, circuit):
    """QALM (test_greedy_k_ablation / test_qalm) and Quartz (test_original) logs."""
    series, kevents, greedy = [], [], []
    extra = {"post_greedy": None, "seed_line": None, "timeout_lines": 0}
    final_lines = None
    pg_lines = None
    pg_t = None
    prefix = f"[{circuit}] "
    with open_text(stdout_path) as f:
        for line in f:
            if final_lines is not None:
                final_lines.append(line)
                continue
            if pg_lines is not None:
                if line.startswith("End post-greedy graph."):
                    c = count_text("".join(pg_lines))
                    extra["post_greedy"] = {"t_s": pg_t, **c}
                    _append(series, pg_t, c["total"], c["cx"])
                    pg_lines = None
                else:
                    pg_lines.append(line)
                continue
            line_s = line.rstrip("\n")
            if line_s.startswith(prefix):
                m = _PROGRESS.match(line_s)
                if m and m.group("name") == circuit:
                    _append(series, float(m.group("t")), int(round(float(m.group("cost")))),
                            int(m.group("cx")))
                    continue
                m = _KSTART.match(line_s)
                if m and m.group("name") == circuit:
                    kevents.append((int(m.group("k")), float(m.group("t"))))
                continue
            if line_s == "Optimized graph:":
                final_lines = []
                continue
            if line_s == "Post-greedy graph:":
                pg_lines = []
                continue
            m = _POST_GREEDY_T.match(line_s)
            if m:
                pg_t = float(m.group("t"))
                continue
            m = _GREEDY_TIMEOUT.match(line_s)
            if m:
                # the message prints the budget, not when the pass stopped, so the post-greedy
                # circuit keeps its measured time (same strict rule as every other tool)
                extra["greedy_hit_timeout"] = True
                continue
            m = _GREEDY.match(line_s)
            if m:
                greedy.append((m.group("pass"), float(m.group("t")),
                               float(m.group("a")), float(m.group("b"))))
                continue
            if line_s.startswith("Seed: "):
                extra["seed_line"] = line_s[6:]
            elif "Program terminated" in line_s:
                extra["timeout_lines"] += 1
    final = None
    if final_lines:
        extra["final_text"] = "".join(final_lines)
        try:
            final = count_text(extra["final_text"])
        except ValueError as e:
            extra["final_parse_error"] = str(e)
    return {"series": series, "kevents": kevents, "greedy": greedy, "final": final, "extra": extra}


def parse_guoq(out_dir, result):
    """GUOQ / QUESO: JSON lines in tool.out; final circuit = latest_sol_*.qasm."""
    series = []
    extra = {"config": None, "original": None, "java_oom": False}
    log = os.path.join(out_dir, "tool.out")
    if os.path.exists(log) or os.path.exists(log + ".gz"):
        with open_text(log) as f:
            for line in f:
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "job_info" in d:
                    extra["config"] = d
                elif "original_total" in d:
                    extra["original"] = d
                elif "best_circuit_size" in d and "seconds_elapsed" in d:
                    _append(series, float(d["seconds_elapsed"]), int(d["best_circuit_size"]),
                            int(d["best_size_2q"]))
    err = os.path.join(out_dir, "tool.err")
    if os.path.exists(err):
        with open(err, errors="replace") as f:
            extra["java_oom"] = "OutOfMemoryError" in f.read()
    final = None
    qasm = os.path.join(out_dir, result.get("output_qasm", ""))
    if result.get("output_qasm") and os.path.exists(qasm):
        with open(qasm) as f:
            text = f.read()
        # GUOQ rewrites latest_sol in place and always ends it with a blank line; a file
        # without it was cut mid-write (e.g. by the SIGTERM at the budget)
        if not text.endswith("\n\n"):
            extra["final_truncated"] = True
        else:
            try:
                final = count_text(text)
            except ValueError as e:
                extra["final_parse_error"] = str(e)
    return {"series": series, "kevents": [], "greedy": [], "final": final, "extra": extra}


def parse_voqc(out_dir, result):
    final = None
    extra = {}
    qasm = os.path.join(out_dir, result.get("output_qasm", ""))
    if result.get("output_qasm") and os.path.exists(qasm):
        with open(qasm) as f:
            try:
                final = count_text(f.read())
            except ValueError as e:
                extra["final_parse_error"] = str(e)
    js = os.path.join(out_dir, os.path.dirname(result.get("output_qasm", "")), "results_none_none.json")
    if os.path.exists(js):
        with open(js) as f:
            extra["voqc_json"] = json.load(f)
    return {"series": [], "kevents": [], "greedy": [], "final": final, "extra": extra}
