"""Apply the regenerated results to a checkout of the QALM paper, in place.

Inputs
    --paper DIR        a checkout of the paper (main.tex, texts/, figtex/,
                       figure/).  It is modified in place: point it at a fresh
                       branch or at a copy.
    --out-dir DIR      the generated outputs, all in one directory:
                         numbers.tex, numbers.json          numbers.py
                         averages_table.tex,
                         eval_end_to_end.tex                tables.py
                         comparison_3x3.pdf, win_rate.png,
                         gate_scatter.pdf,
                         fig_percircuit.json                fig_percircuit.py
                         greedy_k_reduction.pdf,
                         single_k_gk0.pdf, single_k_gk2.pdf,
                         N_pool_arith_mean_3600s_plain.pdf,
                         N_branch_arith_mean_3600s_plain.pdf,
                         curves_numbers.json                fig_curves.py
    --qalm-config a|b  QALM configuration of the main comparison.  It must be
                       the one the outputs were generated with (numbers.json
                       meta, fig_percircuit.json, and the averages table must
                       agree with numbers.tex).  Default: numbers.json's.

What it changes in the paper
    1. Copies the 8 figures into figure/ and averages_table.tex,
       eval_end_to_end.tex and numbers.tex into figtex/.
    2. main.tex: ``\\input{figtex/numbers}`` once, before
       ``\\begin{document}``.
    3. Replaces every experimental number of the compiled text (texts/*.tex
       outside comment environments and %-comments, figtex captions) by its
       ``\\num...{}`` macro, following the reviewed table SUBS below.  Each
       entry quotes the paper text around its number(s) and must match exactly
       one place of the compiled text of its file; there is no global search
       and replace.
    4. Review blocks (REVIEW_BLOCKS): the experimental-setup paragraph and the
       Docker sentence are replaced by a description of the re-run's
       methodology, and the E5 figure caption gets a sentence on the seed
       mean and min..max band.  Each block is framed by
       ``% REPRO-REVIEW-BEGIN <id> sha=<hash> -- ...`` and
       ``% REPRO-REVIEW-END <id>`` lines for the user to review.  The hash is
       that of the generated text: a block whose text no longer has that hash
       was edited by hand and is kept (unless --refresh-blocks).
    5. A ``% REPRO-CHECK: <claim id> is now <true|false|undetermined>``
       comment line right above every sentence whose qualitative truth is a
       claim of numbers.json (MARKERS).  These sentences are never reworded;
       the report lists them, false and undetermined ones first.

Report (stdout; also as JSON with --report FILE): inputs and checks, copied
files, substitutions (applied / already applied), review blocks, REPRO-CHECK
markers, and every number of the compiled text that was not replaced: those
kept on purpose (KEEP, with the reason) and any others, listed for review.

Safety: all edits are computed in memory first.  Nothing is written if a
SUBS template or marker anchor does not match exactly once, a used macro is
not defined in numbers.tex, the brace balance of a file changes, the inputs
disagree with each other or with --qalm-config, or numbers.json is
incomplete (provisional values) and --allow-partial is not given.  The
figures must come from the same results as numbers.json (fig_percircuit.json
column means, curves_numbers.json input hashes, strict mode, every figure
written) and from the paper's budgets (3600 s / 60 s); with --allow-partial
a mismatch is only a warning.  Running
the script again on its own output changes nothing (idempotent): applied
substitutions are recognised, markers are removed and re-inserted, and
review blocks are regenerated only while unedited.

--dry-run prints a unified diff of the text files (and lists the binary
files that would change) instead of writing anything.

Exit status: 0 ok, 1 refused (nothing written), 2 written/diffed but with
numbers left for review.

CLI (from the repo root):
    python3 repro/analysis/paper_patch.py --paper <paper checkout> \\
        --out-dir repro/analysis/out [--qalm-config a|b] [--dry-run] \\
        [--report patch_report.json] [--allow-partial] [--refresh-blocks]
"""

import argparse
import difflib
import fnmatch
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)

FIGURES = [
    "comparison_3x3.pdf",
    "win_rate.png",
    "gate_scatter.pdf",
    "greedy_k_reduction.pdf",
    "single_k_gk0.pdf",
    "single_k_gk2.pdf",
    "N_pool_arith_mean_3600s_plain.pdf",
    "N_branch_arith_mean_3600s_plain.pdf",
]
FIGTEX = ["averages_table.tex", "eval_end_to_end.tex", "numbers.tex"]
GENERATED_TEX = {"figtex/" + f for f in FIGTEX}
EVAL = "texts/5-eval.tex"
INTRO = "texts/1-introduction.tex"
ABSTRACT = "texts/0-abstract.tex"
CONCL = "texts/7-conclusion.tex"
NUMBERS_INPUT = "\\input{figtex/numbers}"
MARKER_PREFIX = "% REPRO-CHECK: "
BLOCK_BEGIN = "% REPRO-REVIEW-BEGIN "
BLOCK_END = "% REPRO-REVIEW-END "

# ---------------------------------------------------------------------------
# Reviewed mapping: old literal -> macro, with the paper text around it.
#
# Template syntax: the text exactly as in the paper source, except that any
# run of whitespace matches any run of whitespace (so line breaks may differ)
# and each ``[[old=>macro]]`` slot is one number.  A slot matches the old
# literal (it is then replaced by ``\macro{}``) or the macro itself (already
# applied).  Every template must match exactly once in the compiled text of
# its file.  Line numbers in the comments refer to the paper as of the
# 2026-10-30 audit.
# ---------------------------------------------------------------------------
SUBS = [
    # abstract
    (
        "abstract-suite",
        ABSTRACT,
        r"In a comprehensive evaluation across [[248=>numSuiteCircuits]] circuits,",
    ),  # :38
    (
        "abstract-win-fid",
        ABSTRACT,
        r"the strongest baseline on $[[83.9=>numWinFidStrongest]]\%$ of these circuits",
    ),  # :39-40, strongest = lowest fidelity win-or-tie rate (claim checks it)
    # introduction
    (
        "intro-suite",
        INTRO,
        r"Our evaluation on [[248=>numSuiteCircuits]] circuits shows that",
    ),  # :121
    (
        "intro-margin-1h",
        INTRO,
        r"\optName{} achieves [[5.97=>numTotalRedMarginOneHour]]\% more gate"
        r" count reduction than the strongest baseline,",
    ),  # :123-124, strongest = highest mean total reduction
    (
        "intro-margin-1min",
        INTRO,
        r"it still achieves [[4.98=>numTotalRedMarginOneMin]]\% more reduction"
        r" than the strongest baseline.",
    ),  # :125-126
    (
        "intro-win-fid-guoq2q",
        INTRO,
        r"still leads to higher or equal fidelity on [[83.9=>numWinFidGuoqTwoQ]]\%"
        r" of circuits.",
    ),  # :136, the sentence names GUOQ (2Q mode)
    # evaluation: setup
    (
        "setup-suite",
        EVAL,
        r"benchmarks consisting of [[248=>numSuiteCircuits]] circuits compiled by",
    ),  # :27
    (
        "setup-source-suite",
        EVAL,
        r"This circuit set originally consisted of [[250=>numSuiteSourceCircuits]]"
        r" circuits;",
    ),  # :28
    (
        "setup-span",
        EVAL,
        r"The [[248=>numSuiteCircuits]] circuits span [[five=>numGateCountOrders]]"
        r" orders of magnitude in gate count.",
    ),  # :40
    # evaluation: main results (tab:avg-comparison)
    (
        "main-guoq2q-cx",
        EVAL,
        r"nearly identical $CX$ reduction ($[[24.61=>numCxRedQalm]]\%$ vs."
        r" $[[25.41=>numCxRedGuoqTwoQ]]\%$)",
    ),  # :116-117
    (
        "main-guoq2q-total",
        EVAL,
        r"dramatically better total gate reduction ($[[52.34=>numTotalRedQalm]]\%$"
        r" vs. $[[-2.63=>numTotalRedGuoqTwoQ]]\%$)",
    ),  # :117-118
    (
        "main-voqc-total",
        EVAL,
        r"similar total gate reductions ($[[52.34=>numTotalRedQalm]]\%$ vs."
        r" $[[46.37=>numTotalRedVoqc]]\%$)",
    ),  # :122
    (
        "main-voqc-cx",
        EVAL,
        r"significantly greater $CX$ reduction ($[[24.61=>numCxRedQalm]]\%$ vs."
        r" $[[4.24=>numCxRedVoqc]]\%$)",
    ),  # :123-124
    (
        "main-fidelity",
        EVAL,
        r"\optName{} (average $[[0.3864=>numFidQalm]]$) outperforms both \voqc{}"
        r" ($[[0.3428=>numFidVoqc]]$) and \guoq{} in two-qubit mode"
        r" ($[[0.3475=>numFidGuoqTwoQ]]$)",
    ),  # :124-126
    # evaluation: optimization efficiency
    (
        "efficiency-factor",
        EVAL,
        r"reducing the time budget by a factor of $[[60=>numBudgetFactor]]$",
    ),  # :139
    # evaluation: Quartz's benchmark (tab:eval_end_to_end)
    (
        "quartz26-suite",
        EVAL,
        r"we also compare with \quartz{} on the [[26=>numQuartzSuiteCircuits]]"
        r" benchmark circuits used by \quartz{}.",
    ),  # :150-151
    (
        "quartz26-all",
        EVAL,
        r"optimizer on all [[26=>numQuartzSuiteCircuits]] benchmark circuits.",
    ),  # :156
    (
        "quartz26-except",
        EVAL,
        r"on all of them except for [[one=>numETEQalmNotBestWord]] circuit compared"
        r" to \quartz with preprocessing.",
    ),  # :158-159
    (
        "quartz26-all-but-one",
        EVAL,
        r"still matches or outperforms it on all but [[one=>numETEQalmWorseQuartzPPWord]]"
        r" circuit.",
    ),  # :171-172
    (
        "quartz26-delta-voqc",
        EVAL,
        r"reduces gate count by an additional $[[7.3=>numGeoDeltaVoqc]]\%$ on"
        r" average.",
    ),  # :160
    (
        "quartz26-delta-search",
        EVAL,
        r"outperforms \quartz{} (without preprocessing) by"
        r" $[[17.8=>numGeoDeltaQuartzNoPP]]\%$, \queso{} by"
        r" $[[10.4=>numGeoDeltaQueso]]\%$, \guoq{} by $[[5.1=>numGeoDeltaGuoq]]\%$,"
        r" and \quartz{} (with preprocessing) by $[[4.6=>numGeoDeltaQuartzPP]]\%$.",
    ),  # :161-164
    # evaluation: greedy mode (fig:greedy_k)
    (
        "greedy-finals",
        EVAL,
        r"([[32.80=>numGreedyFinalGkZero]]\% without greedy mode versus"
        r" [[33.27=>numGreedyFinalGkTwo]]\% with greedy mode for $k\le 2$)",
    ),  # :321-322
    (
        "greedy-t-gk0",
        EVAL,
        r"\optName{} takes [[968=>numGreedyTimeToThresholdGkZero]] seconds to reach"
        r" an average gate count reduction of [[32=>numGreedyThreshold]]\% on this"
        r" benchmark suite.",
    ),  # :323-325
    (
        "greedy-t-gk1",
        EVAL,
        r"reaches the same threshold in [[89=>numGreedyTimeToThresholdGkOne]]"
        r" seconds, an [[11=>numGreedySpeedupGkOne]]$\times$ speedup.",
    ),  # :326
    (
        "greedy-phase-gk1-gk2",
        EVAL,
        r"the greedy phase takes [[5.5=>numGreedyPhaseMeanGkOne]] seconds on"
        r" average. For $k \le 2$, it takes [[82=>numGreedyPhaseMeanGkTwo]] seconds"
        r" on average.",
    ),  # :329-330
    (
        "greedy-phase-gk3",
        EVAL,
        r"pushes the greedy phase to [[724=>numGreedyPhaseMeanGkThree]] seconds on"
        r" average, and [[two=>numGreedyNotFinishedGkThreeWord]] of the"
        r" [[26=>numQuartzSuiteCircuits]] circuits"
        r" ([[\texttt{gf2\^{}9\_mult} and \texttt{gf2\^{}10\_mult}"
        r"=>numGreedyNotFinishedGkThreeList]])",
    ),  # :330-332
    (
        "greedy-k3-final",
        EVAL,
        r"The resulting final reduction ([[33.12=>numGreedyFinalGkThree]]\%) is also"
        r" slightly below the $k \le 2$ setting ([[33.27=>numGreedyFinalGkTwo]]\%)",
    ),  # :334-335
    # evaluation: exploration depth (fig:single_k)
    (
        "singlek-caption",
        EVAL,
        r"(no depth advancement), averaged over [[26=>numQuartzSuiteCircuits]]"
        r" benchmark circuits.}",
    ),  # :357-358 (caption)
    (
        "singlek-gk0",
        EVAL,
        r"decreases monotonically with $k$, from [[32.83=>numFixedKGkZeroKOne]]\% at"
        r" $k=1$ down to [[31.46=>numFixedKGkZeroKFive]]\% at $k=5$,",
    ),  # :367-368
    (
        "singlek-gk2-best",
        EVAL,
        r"the ordering shifts: $k=[[3=>numFixedKGkTwoBestK]]$ becomes the best"
        r" single fixed depth ([[33.11=>numFixedKGkTwoBest]]\%)",
    ),  # :374-375
    (
        "singlek-advancing",
        EVAL,
        r"matches the advancing-$k$ baseline ([[33.27=>numFixedKGkTwoAdvancing]]\%)",
    ),  # :377-378
    # evaluation: exploration strategy (fig:exploration_ablation)
    (
        "e5-caption-suite",
        EVAL,
        r"average gate count reduction, over [[26=>numQuartzSuiteCircuits]]"
        r" benchmark circuits with a one-hour timeout.",
    ),  # :409-410 (caption)
    (
        "e5-npool-range",
        EVAL,
        r"the final average reduction ranges from [[33.03=>numNpoolMin]]\% to"
        r" [[33.16=>numNpoolMax]]\% (a spread of [[0.13=>numNpoolSpread]]\%)",
    ),  # :418-419, E5 values are means over seeds 0-2
    (
        "e5-nbranch-1",
        EVAL,
        r"the average reduction drops to [[31.33=>numNbranchOne]]\%.",
    ),  # :431
    (
        "e5-nbranch-2-3",
        EVAL,
        r"quality climbs to [[32.75=>numNbranchTwo]]\% at $N_\text{branch}=2$ and"
        r" peaks at [[33.37=>numNbranchThree]]\% at $N_\text{branch}=3$.",
    ),  # :433-434
    (
        "e5-nbranch-4-5",
        EVAL,
        r"slightly degrading quality ([[33.30=>numNbranchFour]]\% and"
        r" [[33.21=>numNbranchFive]]\% at $N_\text{branch}=4$ and $5$)",
    ),  # :437-438
]

# ---------------------------------------------------------------------------
# REPRO-CHECK markers: (claim id, file, anchor).  The anchor is the start of
# the sentence (or clause) whose truth the claim checks; the marker goes on
# its own line right above the line where the anchor starts.  Anchors use
# the template syntax above and match the patched text.
# ---------------------------------------------------------------------------
MARKERS = [
    ("intro_outperforms_all_total", ABSTRACT, r"Crucially, our results demonstrate"),
    ("every_column_all_248", ABSTRACT, r"In a comprehensive evaluation across"),
    (
        "abstract_strongest_baseline_fidelity",
        ABSTRACT,
        r"In a comprehensive evaluation across",
    ),
    (
        "intro_outperforms_all_total",
        INTRO,
        r"Our evaluation on [[248=>numSuiteCircuits]] circuits shows",
    ),
    ("intro_margin_positive_qalm", INTRO, r"Given the same time budget (one hour),"),
    (
        "intro_margin_positive_qalm_1min",
        INTRO,
        r"Given the same time budget (one hour),",
    ),
    ("intro_nearly_every_circuit", INTRO, r"Compared with \queso{} (in both two-qubit"),
    ("guoq2q_cx_on_par", INTRO, r"Although the two-qubit gate reduction is on par"),
    (
        "guoq2q_total_dramatically_better",
        INTRO,
        r"Although the two-qubit gate reduction is on par",
    ),
    (
        "abstract_strongest_baseline_fidelity",
        INTRO,
        r"Although the two-qubit gate reduction is on par",
    ),
    ("every_column_all_248", EVAL, r"To evaluate our framework, we employ"),
    (
        "five_orders_of_magnitude",
        EVAL,
        r"The [[248=>numSuiteCircuits]] circuits span",
    ),
    ("cpu_model", EVAL, r"All experiments were conducted on servers"),  # setup block
    ("memory_never_limiting", EVAL, r"Each run may use up to"),  # setup block
    (
        "majority_strict_fidelity",
        EVAL,
        r"\optName{} achieves higher fidelity than all four baselines",
    ),
    (
        "guoq2q_cx_on_par",
        EVAL,
        r"Against \guoq{} in two-qubit gate optimization mode,",
    ),
    (
        "guoq2q_total_dramatically_better",
        EVAL,
        r"Against \guoq{} in two-qubit gate optimization mode,",
    ),
    ("guoq2q_negative_on_many", EVAL, r"\guoq{}'s resynthesis frequently introduces"),
    ("qalm_never_negative", EVAL, r"whereas \optName{} avoids this entirely."),
    ("voqc_similar_total", EVAL, r"Against \voqc{}, the situation is reversed:"),
    ("voqc_much_greater_cx", EVAL, r"Against \voqc{}, the situation is reversed:"),
    ("fidelity_beats_voqc_guoq2q", EVAL, r"In terms of fidelity, \optName{} (average"),
    ("table_qalm_best_fidelity", EVAL, r"In terms of fidelity, \optName{} (average"),
    ("table_qalm_best_total_red", EVAL, r"In terms of fidelity, \optName{} (average"),
    ("remaining_baselines_all_three", EVAL, r"Against the remaining baselines,"),
    (
        "one_minute_degrades_lt_1pct",
        EVAL,
        r"As shown in \Cref{tab:avg-comparison}, reducing the time budget",
    ),
    (
        "one_minute_beats_every_baseline",
        EVAL,
        r"As shown in \Cref{tab:avg-comparison}, reducing the time budget",
    ),
    (
        "ete_lowest_all_but_one",
        EVAL,
        r"\optName{} achieves the lowest or equally lowest",
    ),
    (
        "ete_outperforms_by_positive",
        EVAL,
        r"Compared to the purely rule-based \voqc{},",
    ),
    (
        "ete_matches_quartz_pp_all_but_one",
        EVAL,
        r"In addition, \quartz{} with preprocessing takes advantage",
    ),
    (
        "ete_consistently_better",
        EVAL,
        r"These results demonstrate that the interleaved",
    ),
    (
        "ete_subsumes_preprocessing",
        EVAL,
        r"These results demonstrate that the interleaved",
    ),
    (
        "ablations_all_26",
        EVAL,
        r"The following ablation studies use the same circuits as",
    ),
    (
        "table_qalm_is_ablation_default",
        EVAL,
        r"The following ablation studies use the same circuits as",
    ),
    (
        "greedy_final_diff_lt_half",
        EVAL,
        r"While the final gate count reductions differ",
    ),
    ("greedy_gk2_above_gk0", EVAL, r"While the final gate count reductions differ"),
    (
        "greedy_k1_reaches_threshold_much_faster",
        EVAL,
        r"While the final gate count reductions differ",
    ),
    (
        "greedy_phase_grows_with_k",
        EVAL,
        r"However, it is not always beneficial to extend greedy mode",
    ),
    (
        "greedy_k3_some_not_finished",
        EVAL,
        r"Extending greedy mode to $k \le 3$ pushes",
    ),
    ("greedy_k3_below_k2", EVAL, r"The resulting final reduction ("),
    (
        "greedy_k2_best_of_four",
        EVAL,
        r"With a time limit of one hour, enabling greedy mode",
    ),
    (
        "fixed_k_gk0_monotone",
        EVAL,
        r"Without the greedy mode (\Cref{fig:single_k_gk0}), the final average",
    ),
    (
        "fixed_k_gk2_k3_best",
        EVAL,
        r"With the greedy mode enabled for $k \le 2$ and a fixed $k$ after that",
    ),
    (
        "fixed_k_ordering_shifts",
        EVAL,
        r"With the greedy mode enabled for $k \le 2$ and a fixed $k$ after that",
    ),
    (
        "fixed_k_gk0_none_matches_advancing",
        EVAL,
        r"Note that no single fixed $k$ matches",
    ),
    (
        "fixed_k_gk2_none_matches_advancing",
        EVAL,
        r"Note that no single fixed $k$ matches",
    ),
    (
        "npool_no_meaningful_effect",
        EVAL,
        r"$N_\text{pool}$ has little effect on final quality,",
    ),  # caption
    (
        "nbranch_sweet_spot_at_3",
        EVAL,
        r"$N_\text{pool}$ has little effect on final quality,",
    ),  # caption
    (
        "nbranch_sweet_spot_resolved",
        EVAL,
        r"$N_\text{pool}$ has little effect on final quality,",
    ),  # caption
    (
        "npool_no_meaningful_effect",
        EVAL,
        r"Pool size $N_\text{pool}$ has no meaningful effect",
    ),
    (
        "nbranch_material_effect",
        EVAL,
        r"Branch factor $N_\text{branch}$, in contrast, materially",
    ),
    (
        "nbranch_sweet_spot_at_3",
        EVAL,
        r"Branch factor $N_\text{branch}$, in contrast, materially",
    ),
    (
        "nbranch_sweet_spot_resolved",
        EVAL,
        r"Branch factor $N_\text{branch}$, in contrast, materially",
    ),
    (
        "nbranch_climbs_to_3",
        EVAL,
        r"Raising $N_\text{branch}$ introduces the diversity",
    ),
    (
        "default_config_one_value",
        EVAL,
        r"Raising $N_\text{branch}$ introduces the diversity",
    ),
    ("nbranch_degrades_after_3", EVAL, r"Beyond this point, the additional branches"),
    (
        "default_config_justified",
        EVAL,
        r"We therefore use $N_\text{pool}=1$ and $N_\text{branch}=3$ as the default",
    ),
    (
        "qalm_is_described_default",
        EVAL,
        r"We therefore use $N_\text{pool}=1$ and $N_\text{branch}=3$ as the default",
    ),
    (
        "qalm_is_described_default",
        "texts/3-method.tex",
        r"For our main search loop, we use the constant",
    ),
    (
        "gate_scatter_strongest_baseline",
        "figtex/gate_scatter.tex",
        r"The left panel shows the distribution before optimization,",
    ),
    ("intro_outperforms_all_total", CONCL, r"Our evaluation confirms that this design"),
    (
        "conclusion_one_minute_beats_search_based",
        CONCL,
        r"Our evaluation confirms that this design",
    ),
]

# ---------------------------------------------------------------------------
# Review blocks: (id, file, template, mode, description).  mode "replace"
# replaces the whole template match; mode "insert" inserts the block at the
# [[@]] position of the template.  The text comes from block_text().
# ---------------------------------------------------------------------------
REVIEW_BLOCKS = [
    (
        "setup-docker",
        EVAL,
        r"We execute \guoq and \queso within the Docker environment provided by"
        r" their respective authors.",
        "replace",
        "which tools run in the authors' Docker image",
    ),
    (
        "setup-method",
        EVAL,
        r"All experiments were conducted on servers equipped with AMD EPYC 7763"
        r" processors. Each optimizer is given one CPU core, 32 GiB of RAM, and a"
        r" one-hour time limit for each run.",
        "replace",
        "methodology of the re-run (cores, CPU mode, memory, seeds, result rule)",
    ),
    (
        "e5-caption",
        EVAL,
        r"while $N_\text{branch}$ exhibits a sweet spot at $N_\text{branch}=3$.[[@]]}",
        "insert",
        "E5 curves are seed means with a min..max band",
    ),
]

SETUP_DOCKER = r"""We execute \guoq{}, \queso{}, and \voqc{} in the Docker image provided
by the authors of \guoq{}~\cite{xu2024optimizing}, which contains all
three tools; \optName{} and \quartz{} run natively on the same servers."""

SETUP_METHOD_COMMON = r"""All experiments were conducted on servers equipped
with AMD EPYC 7763 processors. Each run is pinned to one physical CPU
core whose second hardware thread (SMT sibling) is left idle, and up to
\numMaxConcurrent{} runs share a server. CPU frequency boost is
disabled, so that the speed of a core does not depend on the load of
the server. Each run has a one-hour time limit.
Each run may use up to \numMemCapGiB{}~GiB of memory: to fit more runs
on a server, a run starts with a smaller memory limit derived from its
expected peak usage and is restarted with the next larger limit, up to
\numMemCapGiB{}~GiB, if it reaches its limit before the time limit ends.
Unlike in the authors' scripts, the resynthesis server of \guoq{} uses
a single worker on the run's core and does not busy-wait.
All randomized optimizers use seed~0; only the exploration-strategy
study (\Cref{sec:eval_explore}) repeats each configuration with seeds
0--2.
The result of a run is the best circuit the optimizer reported within
its time limit. A run that ended abnormally (for example, by exceeding
its memory limit) counts with the best circuit it reported before, or
with the original circuit if it reported none; the table captions give
the number of such runs. All averages over the benchmark suite,
including those of \voqc{}, are taken over all \numSuiteCircuits{}
circuits."""

SETUP_METHOD_QALM = {
    "a": r"""Unless stated otherwise, \optName{} runs in its default configuration
($N_\text{pool}=1$, $N_\text{branch}=3$, and greedy mode for $k\le 2$;
see \Cref{sec:eval_explore}).""",
    "b": r"""For the main comparison (\Cref{tab:avg-comparison,tab:eval_end_to_end}),
\optName{} runs with $N_\text{pool}=1$ and $N_\text{branch}=1$: greedy
mode for $k\le 2$, then a fixed exploration depth $k=3$ in which the
intermediate circuits of each branch are also enqueued. The ablation
studies start from the default configuration ($N_\text{pool}=1$,
$N_\text{branch}=3$, advancing $k$; \Cref{sec:eval_explore}).""",
}

E5_CAPTION = r"""    Each line is the mean over seeds 0--2, and the shaded band spans
    the minimum to the maximum over these seeds."""


def block_text(block_id, qalm_config):
    """The generated text of one review block (no trailing newline)."""
    if block_id == "setup-docker":
        return SETUP_DOCKER
    if block_id == "setup-method":
        return SETUP_METHOD_COMMON + "\n" + SETUP_METHOD_QALM[qalm_config]
    if block_id == "e5-caption":
        return E5_CAPTION
    raise KeyError(block_id)


# ---------------------------------------------------------------------------
# Numbers of the compiled text that stay on purpose: (file glob, regex,
# reason).  A number inside a match of a rule for its file is "kept"; every
# other number is reported for review.
# ---------------------------------------------------------------------------
METHOD_FILES = [
    "texts/2-background.tex",
    "texts/3-method.tex",
    "texts/4-implementation.tex",
    "figtex/top_level_algo.tex",
    "figtex/explore_random.tex",
    "figtex/explore_greedy.tex",
    "figtex/search_advantage.tex",
]
KEEP = [
    (
        "*",
        r"\\texttt\{(?:[^{}]|\{[^{}]*\})*\}",
        "identifier or command-line flag in \\texttt (circuit name, tool option)",
    ),
    (
        "*",
        r"\$[^$]*?(?:\bk\b|N_|\\in\\\{)[^$]*?\$",
        "configuration parameter value (k, N_pool, N_branch), not a result",
    ),
    (EVAL, r"\$\\\{1,\s*2,\s*3,\s*4,\s*5\\\}\$", "parameter grid of the E5 study"),
    (EVAL, r"=4\$ and \$5\$", "parameter value (N_branch=5)"),
    (EVAL, r"setting them to 1\)", "algorithm constant (greedy passes)"),
    (EVAL, r"holds at most\s+one circuit", "algorithm description (N_branch=1)"),
    (EVAL, r"[fn]_\{[12]q\}", "symbol of the fidelity model"),
    (EVAL, r"f_\{[12]q\}\s*=\s*0\.99+", "constant of the fidelity model (0.999, 0.99)"),
    (EVAL, r"\(v0\.3\.2\)", "tool version"),
    (EVAL, r"AMD EPYC\s+7763", "CPU model (claim cpu_model checks the host snapshots)"),
    ("*", r"seeds?(?:~|\s+)0(?:--2)?", "seed values (setup)"),
    (
        EVAL,
        r"by less than \$1\\%\$",
        "threshold of claim one_minute_degrades_lt_1pct (measured drops:"
        " \\numOneMinFidDropRel, \\numOneMinCxRedDrop, \\numOneMinTotalRedDrop)",
    ),
    (
        EVAL,
        r"by\s+less than 0\.5\\%",
        "threshold of claim greedy_final_diff_lt_half (measured: \\numGreedyFinalDiff)",
    ),
]
for _f in METHOD_FILES:
    KEEP += [
        (
            _f,
            r"\$[^$]*\$",
            "math in the method/background text (symbolic, illustrative)",
        ),
        (_f, r"\\begin\{yquant\}.*?\\end\{yquant\}", "circuit diagram code"),
        (_f, r"either 0 or\s+1\.|both 0 and 1", "background: bit values"),
        (_f, r"more than 2,000 circuits|top 1,000 circuits", "algorithm constant"),
        (_f, r"at most 3 qubits|at most 5 gates", "ECC set parameters"),
        (_f, r"to 1, we maintain", "algorithm constant (greedy passes)"),
        (_f, r"\b(?:one|two|three)\b", "number word in the method/background prose"),
    ]

# Arguments that are not prose (blanked before looking for numbers).
NON_PROSE = re.compile(
    r"\\(?:cite[pt]?|ref|Cref|cref|eqref|autoref|label|input|include|bibliography"
    r"|bibliographystyle|url|includegraphics|vspace|hspace|setlength)\*?"
    r"(?:\[[^\]]*\])?\{[^{}]*\}(?:\{[^{}]*\})?"
    r"|\\(?:begin|end)\{[^{}]*\}(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})?"
    r"|-?\d*\.?\d+\s*\\(?:linewidth|columnwidth|textwidth|baselineskip)"
    r"|\b\d*\.?\d+\s*(?:cm|mm|pt|em|ex)\b"
)
NUMBER = re.compile(r"(?<![A-Za-z\\])-?\d+(?:[.,]\d+)*")
WORD_NUMBER = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    r"(?=\s+(?:of\s+the\b|circuits?\b|orders?\s+of\b))"
)
EXPERIMENTAL_AFTER = re.compile(
    r"\s*(?:\\%|%|\$?\\times|\s*seconds\b|s\b|~?GiB\b|GB\b|\s*circuits?\b"
    r"|\s*hours?\b|\s*minutes?\b|\s*gates\b|\s*runs?\b)"
)


class Refused(Exception):
    """The inputs or the paper do not allow a safe patch; nothing is written."""


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def mask(text):
    """Same-length copy of ``text`` with comments replaced by NUL characters.

    Masks %-comments (an unescaped % to the end of its line) and
    ``\\begin{comment}`` ... ``\\end{comment}`` environments; newlines stay.
    """
    lines = text.split("\n")
    out = []
    for line in lines:
        cut = None
        for j, ch in enumerate(line):
            if ch == "%":
                k, n = j - 1, 0
                while k >= 0 and line[k] == "\\":
                    n, k = n + 1, k - 1
                if n % 2 == 0:
                    cut = j
                    break
        out.append(line if cut is None else line[:cut] + "\0" * (len(line) - cut))
    masked = list("\n".join(out))
    joined = "".join(masked)
    for m in re.finditer(r"\\begin\{comment\}.*?\\end\{comment\}", joined, re.S):
        for p in range(m.start(), m.end()):
            if masked[p] != "\n":
                masked[p] = "\0"
    return "".join(masked)


def brace_balance(text):
    """(open - close braces, lowest running depth), unescaped and outside comments."""
    t = re.sub(r"\\[{}]", "", re.sub(r"\\\\", "", mask(text)))
    depth = lowest = 0
    for ch in t:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            lowest = min(lowest, depth)
    return depth, lowest


SLOT = re.compile(r"\[\[(.+?)=>(num[A-Za-z]+)\]\]|\[\[@\]\]", re.S)


def _lit(s):
    parts = []
    for tok in re.split(r"(\s+)", s):
        if tok:
            parts.append(r"\s+" if tok.isspace() else re.escape(tok))
    return "".join(parts)


def macro_rx(name):
    return re.escape("\\" + name) + r"(?:\{\}|(?![A-Za-z]))"


def compile_template(template):
    """(regex, slots, has_insert_point) for one template (see SUBS)."""
    rx, slots, pos, insert = [], [], 0, False
    for m in SLOT.finditer(template):
        rx.append(_lit(template[pos : m.start()]))
        if m.group(0) == "[[@]]":
            rx.append("(?P<at>)")
            insert = True
        else:
            old, name = m.groups()
            rx.append(f"(?P<s{len(slots)}>{_lit(old)}|{macro_rx(name)})")
            slots.append((old, name))
        pos = m.end()
    rx.append(_lit(template[pos:]))
    return re.compile("".join(rx)), slots, insert


def find_unique(template, text, what, allow_missing=False):
    """The single match of ``template`` in the compiled part of ``text``.

    No match returns (None, slots) if ``allow_missing``; several matches are
    always an error (the template is ambiguous).
    """
    rx, slots, _ = compile_template(template)
    hits = list(rx.finditer(mask(text)))
    if not hits and allow_missing:
        return None, slots
    if len(hits) != 1:
        raise Refused(
            f"{what}: template matches {len(hits)} times in the compiled text"
            f" (must be exactly once): {template[:90]!r}"
        )
    return hits[0], slots


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def macro_uses(text):
    return re.findall(r"\\(num[A-Z][A-Za-z]*)", mask(text))


def defined_macros(numbers_tex):
    return re.findall(r"\\newcommand\{\\(num[A-Za-z]+)\}", numbers_tex)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def load_inputs(out_dir, qalm_config, allow_partial):
    """Load and cross-check the generated outputs; return (numbers, files, info)."""
    missing = [
        f
        for f in FIGURES + FIGTEX + ["numbers.json"]
        if not os.path.isfile(os.path.join(out_dir, f))
    ]
    if missing:
        raise Refused(f"missing in --out-dir {out_dir}: {', '.join(missing)}")
    numbers = json.loads(read(os.path.join(out_dir, "numbers.json")))
    meta = numbers["meta"]
    cfg = meta.get("qalm_config")
    if qalm_config is None:
        qalm_config = cfg
    if cfg != qalm_config:
        raise Refused(
            f"--qalm-config {qalm_config} but numbers.json was generated with {cfg}"
        )
    pc_path = os.path.join(out_dir, "fig_percircuit.json")
    if os.path.isfile(pc_path):
        pc_cfg = json.loads(read(pc_path)).get("qalm_config")
        if pc_cfg != qalm_config:
            raise Refused(
                f"fig_percircuit.json was generated with QALM config {pc_cfg}, not {qalm_config}"
            )
    else:
        raise Refused(
            "fig_percircuit.json missing: cannot check the figures' QALM configuration"
        )
    tex = read(os.path.join(out_dir, "numbers.tex"))
    names = defined_macros(tex)
    if set(names) != set(numbers["macros"]):
        raise Refused(
            "numbers.tex and numbers.json define different macros (regenerate both)"
        )
    incomplete = [
        k
        for k, m in numbers["macros"].items()
        if m.get("provisional") or m["printed"].endswith("?")
    ]
    info = {
        "qalm_config": qalm_config,
        "numbers_complete": bool(meta.get("complete")),
        "provisional_macros": incomplete,
        "numbers_results": meta.get("results"),
        "runs_csv_sha256": (meta.get("inputs_sha256") or {}).get("runs.csv"),
    }
    if (not meta.get("complete") or incomplete) and not allow_partial:
        raise Refused(
            f"numbers.json is incomplete ({len(incomplete)} provisional or missing macros);"
            " pass --allow-partial to patch anyway (values print with '?' / '??')"
        )
    check_tables(out_dir, numbers)
    sources = check_figure_sources(out_dir, numbers)
    if sources and not allow_partial:
        raise Refused(
            "the figures were not generated from the same results as numbers.json"
            " (or not for the paper's budgets):\n  " + "\n  ".join(sources)
        )
    info["figure_source_problems"] = sources
    return numbers, qalm_config, info


# Budgets the paper text describes ("one-hour", "one-minute" time limits).
PAPER_BUDGET_S, PAPER_SHORT_BUDGET_S = 3600, 60
PC_FIGURES = ["comparison_3x3.pdf", "win_rate.png", "gate_scatter.pdf"]
PC_COLS = {
    "guoq_2q": "GuoqTwoQ",
    "guoq_total": "GuoqTotal",
    "queso_2q": "QuesoTwoQ",
    "queso_total": "QuesoTotal",
    "voqc": "Voqc",
    "quartz": "Quartz",
}


def check_figure_sources(out_dir, numbers):
    """Problems (list of str) between numbers.json and the figure generators.

    fig_percircuit.json: budget, suite, the three figures written, and the
    unrounded column means equal to numbers.json's (same runs).
    curves_numbers.json: present, same input sha256 as numbers.json, strict
    mode, every figure written, budget and threshold as numbers.json.  Also
    numbers.json's budgets must be the paper's (one hour / one minute).
    """
    meta = numbers["meta"]
    macros = numbers["macros"]
    out = []
    budgets = (meta.get("budget_s"), meta.get("short_budget_s"))
    if budgets != (PAPER_BUDGET_S, PAPER_SHORT_BUDGET_S):
        out.append(
            f"numbers.json budgets {budgets}, the paper describes"
            f" ({PAPER_BUDGET_S}, {PAPER_SHORT_BUDGET_S})"
        )
    pc = json.loads(read(os.path.join(out_dir, "fig_percircuit.json")))
    if (pc.get("budget_s"), pc.get("suite")) != (PAPER_BUDGET_S, "S248"):
        out.append(
            f"fig_percircuit.json: budget {pc.get('budget_s')}, suite {pc.get('suite')}"
        )
    unwritten = [f for f in PC_FIGURES if f not in (pc.get("outputs") or [])]
    if unwritten:
        out.append(f"fig_percircuit.json: not written by this run: {unwritten}")
    cols = {"Qalm": pc.get("qalm") or {}}
    cols.update(
        {s: (pc.get("baselines") or {}).get(k) or {} for k, s in PC_COLS.items()}
    )
    for suffix, entry in cols.items():
        means = entry.get("means") or {}
        for stem, key in (
            ("Fid", "fidelity"),
            ("CxRed", "cx_red"),
            ("TotalRed", "total_red"),
        ):
            name = f"num{stem}{suffix}"
            a, b = means.get(key), (macros.get(name) or {}).get("value")
            if (a is None) != (b is None) or (
                a is not None and abs(a - b) > 1e-9 * max(1.0, abs(b))
            ):
                out.append(f"fig_percircuit.json {suffix} {key} = {a}, {name} = {b}")
    cn_path = os.path.join(out_dir, "curves_numbers.json")
    if not os.path.isfile(cn_path):
        out.append("curves_numbers.json missing: cannot check the curve figures")
        return out
    cn = json.loads(read(cn_path))
    cmeta = cn.get("meta") or {}
    if cmeta.get("inputs_sha256") != meta.get("inputs_sha256"):
        out.append(
            "curves_numbers.json was built from other results (inputs_sha256"
            " differs from numbers.json's)"
        )
    if not cmeta.get("strict"):
        out.append("curves_numbers.json: fig_curves.py ran without strict mode")
    if cn.get("figures_not_written"):
        out.append(f"curves_numbers.json: not written: {cn['figures_not_written']}")
    if (cmeta.get("budget_s"), cmeta.get("suite")) != (PAPER_BUDGET_S, "S26"):
        out.append(
            f"curves_numbers.json: budget {cmeta.get('budget_s')}, suite {cmeta.get('suite')}"
        )
    if cmeta.get("threshold_pct") != meta.get("threshold_pct"):
        out.append(
            f"curves_numbers.json threshold {cmeta.get('threshold_pct')} %,"
            f" numbers.json {meta.get('threshold_pct')} %"
        )
    return out


def _cells(row_tex):
    cells = [c.strip() for c in row_tex.split("&")]
    out = []
    for c in cells:
        c = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", c).replace("\\%", "").strip()
        out.append(c)
    return out


def _norm(printed):
    return printed.replace("\\ensuremath{-}", "-").rstrip("?")


def check_tables(out_dir, numbers):
    """The generated tables must show the same values as numbers.tex."""
    macros = numbers["macros"]
    avg = mask(read(os.path.join(out_dir, "averages_table.tex")))
    cols = [
        "Qalm",
        "QalmOneMin",
        "GuoqTwoQ",
        "GuoqTotal",
        "QuesoTwoQ",
        "QuesoTotal",
        "Voqc",
        "Quartz",
    ]
    rows = {
        "Fid": r"Fidelity\s*&(.*?)\\\\",
        "CxRed": r"2Q Gate Reduction \(\\%\)\s*&(.*?)\\\\",
        "TotalRed": r"Total Gate Reduction \(\\%\)\s*&(.*?)\\\\",
    }
    bad = []
    for stem, rx in rows.items():
        m = re.search(rx, avg, re.S)
        if not m:
            raise Refused(f"averages_table.tex: row {stem} not found")
        cells = _cells(m.group(1))
        if len(cells) != len(cols):
            raise Refused(
                f"averages_table.tex: row {stem} has {len(cells)} cells, not {len(cols)}"
            )
        for col, cell in zip(cols, cells):
            name = f"num{stem}{col}"
            if (
                name in macros
                and cell not in ("--", "")
                and "?" not in macros[name]["printed"]
            ):
                if cell != _norm(macros[name]["printed"]):
                    bad.append(
                        f"averages_table {name}: table {cell} vs numbers.tex {macros[name]['printed']}"
                    )
    e2e = mask(read(os.path.join(out_dir, "eval_end_to_end.tex")))
    m = re.search(r"Geo\. Mean Reduction\}\s*&(.*?)\\\\", e2e, re.S)
    if not m:
        raise Refused("eval_end_to_end.tex: geomean row not found")
    cells = _cells(m.group(1))[1:]  # first cell is the Orig. column
    e2e_cols = ["Voqc", "Queso", "Guoq", "QuartzNoPP", "QuartzPP", "Qalm"]
    if len(cells) != len(e2e_cols):
        raise Refused(
            f"eval_end_to_end.tex: geomean row has {len(cells) + 1} cells, not 7"
        )
    for col, cell in zip(e2e_cols, cells):
        name = f"numGeoRed{col}"
        if (
            name in macros
            and cell not in ("--", "")
            and "?" not in macros[name]["printed"]
        ):
            if cell != _norm(macros[name]["printed"]):
                bad.append(
                    f"eval_end_to_end {name}: table {cell} vs numbers.tex {macros[name]['printed']}"
                )
    if bad:
        raise Refused(
            "the tables and numbers.tex disagree (different data or QALM configuration):\n  "
            + "\n  ".join(bad)
        )


# ---------------------------------------------------------------------------
# The patch
# ---------------------------------------------------------------------------
class Patch:
    """In-memory edit of the paper's text files plus the copied outputs."""

    def __init__(
        self, paper, out_dir, numbers, qalm_config, refresh_blocks, lenient=False
    ):
        self.paper = paper
        self.out_dir = out_dir
        self.numbers = numbers
        self.macros = numbers["macros"]
        self.claims = {c["id"]: c for c in numbers["claims"]}
        self.qalm_config = qalm_config
        self.refresh_blocks = refresh_blocks
        self.lenient = lenient
        self.reapply = False  # set once main.tex is known to input numbers.tex
        self.orig = {}
        self.text = {}
        self.binary = {}  # relpath -> (old bytes or None, new bytes)
        self.report = {
            "copied": [],
            "main_tex": None,
            "subs": [],
            "blocks": [],
            "markers": [],
            "claims_without_marker": [],
            "kept_numbers": [],
            "review_numbers": [],
            "macros_used_not_final": [],
            "markers_not_placed": [],
            "plain_text": [],
            "warnings": [],
        }

    def warn(self, msg):
        self.report["warnings"].append(msg)

    # -- file access --------------------------------------------------------
    def get(self, rel):
        if rel not in self.text:
            path = os.path.join(self.paper, rel)
            if not os.path.isfile(path):
                raise Refused(f"paper file not found: {path}")
            self.orig[rel] = read(path)
            self.text[rel] = self.orig[rel]
        return self.text[rel]

    def put(self, rel, text):
        self.get(rel)
        self.text[rel] = text

    def compiled_files(self):
        """Files reached from main.tex through \\input outside comments."""
        seen, todo = [], ["main.tex"]
        while todo:
            rel = todo.pop(0)
            if rel in seen:
                continue
            seen.append(rel)
            for m in re.finditer(r"\\input\{([^}]*)\}", mask(self.get(rel))):
                child = m.group(1).strip()
                if not os.path.splitext(child)[1]:
                    child += ".tex"
                if (
                    os.path.isfile(os.path.join(self.paper, child))
                    and child not in GENERATED_TEX
                ):
                    todo.append(child)
        return seen

    # -- steps --------------------------------------------------------------
    def copy_outputs(self):
        for name in FIGURES:
            self._copy(name, "figure/" + name, binary=True)
        for name in FIGTEX:
            self._copy(name, "figtex/" + name, binary=False)

    def _copy(self, name, rel, binary):
        new = read(os.path.join(self.out_dir, name), "rb")
        path = os.path.join(self.paper, rel)
        old = read(path, "rb") if os.path.isfile(path) else None
        if binary:
            self.binary[rel] = (old, new)
        else:
            self.orig[rel] = old.decode() if old is not None else ""
            self.text[rel] = new.decode()
        state = "unchanged" if old == new else ("new" if old is None else "replaced")
        self.report["copied"].append({"file": rel, "state": state})

    def add_numbers_input(self):
        text = self.get("main.tex")
        masked = mask(text)
        if re.search(r"\\input\{figtex/numbers(?:\.tex)?\}", masked):
            self.report["main_tex"] = "already inputs figtex/numbers"
            return
        hits = [m for m in re.finditer(r"^\\begin\{document\}", masked, re.M)]
        if len(hits) != 1:
            raise Refused("main.tex: \\begin{document} not found exactly once")
        at = hits[0].start()
        add = (
            "% Experimental numbers, generated by repro/analysis/numbers.py\n"
            + NUMBERS_INPUT
            + "\n"
        )
        self.put("main.tex", text[:at] + add + text[at:])
        self.report["main_tex"] = f"added {NUMBERS_INPUT} before \\begin{{document}}"

    def strip_markers(self, files):
        for rel in files:
            text = self.get(rel)
            new = re.sub(
                r"^[ \t]*" + re.escape(MARKER_PREFIX) + r".*\n", "", text, flags=re.M
            )
            if new != text:
                self.put(rel, new)

    def apply_blocks(self):
        for block_id, rel, template, mode, desc in REVIEW_BLOCKS:
            text = self.get(rel)
            content = block_text(block_id, self.qalm_config)
            begin = f"{BLOCK_BEGIN}{block_id} sha={sha(content)} -- {desc}; written by repro/analysis/paper_patch.py, please review"
            end = f"{BLOCK_END}{block_id}"
            rx = re.compile(
                r"^[ \t]*"
                + re.escape(BLOCK_BEGIN + block_id)
                + r" sha=(\w+)[^\n]*\n(.*?)^[ \t]*"
                + re.escape(end)
                + r"[ \t]*$",
                re.M | re.S,
            )
            found = list(rx.finditer(text))
            if len(found) > 1:
                raise Refused(
                    f"{rel}: review block {block_id} appears {len(found)} times"
                )
            entry = {"id": block_id, "file": rel}
            if found:
                m = found[0]
                existing = m.group(2).rstrip("\n")
                if existing == content:
                    entry["state"] = "unchanged"
                elif sha(existing) == m.group(1) or self.refresh_blocks:
                    self.put(
                        rel,
                        text[: m.start()]
                        + begin
                        + "\n"
                        + content
                        + "\n"
                        + end
                        + text[m.end() :],
                    )
                    entry["state"] = (
                        "regenerated"
                        if sha(existing) == m.group(1)
                        else "overwritten (--refresh-blocks)"
                    )
                else:
                    entry["state"] = (
                        "kept: edited by hand (pass --refresh-blocks to regenerate)"
                    )
                entry["line"] = line_of(
                    self.get(rel), self.get(rel).find(BLOCK_BEGIN + block_id)
                )
                self.report["blocks"].append(entry)
                continue
            m, _ = find_unique(
                template, text, f"{rel}: review block {block_id}", self.reapply
            )
            if m is None:
                # neither markers nor the original text: reviewed and unmarked?
                flat = re.sub(r"\s+", " ", mask(text).replace("\0", " "))
                if re.sub(r"\s+", " ", content).strip() in flat:
                    entry["state"] = "accepted (review markers removed by hand)"
                else:
                    entry["state"] = (
                        "skipped: neither the block nor the original text found"
                        " (edited after review?)"
                    )
                    self.warn(f"{rel}: review block {block_id}: {entry['state']}")
                entry["line"] = None
                self.report["blocks"].append(entry)
                continue
            if mode == "replace":
                a, b = m.start(), m.end()
                # keep the line structure: the block occupies whole lines
                lead = "" if a == 0 or text[a - 1] == "\n" else "\n"
                while b < len(text) and text[b] in " \t":
                    b += 1
                trail = "" if b < len(text) and text[b] == "\n" else "\n"
                new = (
                    text[:a]
                    + lead
                    + begin
                    + "\n"
                    + content
                    + "\n"
                    + end
                    + trail
                    + text[b:]
                )
            else:
                at = m.start("at")
                new = (
                    text[:at]
                    + "\n"
                    + begin
                    + "\n"
                    + content
                    + "\n"
                    + end
                    + "\n"
                    + text[at:]
                )
            self.put(rel, new)
            entry["state"] = "inserted"
            entry["line"] = line_of(new, new.find(BLOCK_BEGIN + block_id))
            self.report["blocks"].append(entry)

    def apply_subs(self):
        for sub_id, rel, template in SUBS:
            text = self.get(rel)
            m, slots = find_unique(
                template, text, f"{rel}: substitution {sub_id}", self.reapply
            )
            if m is None:
                present = set(macro_uses(text))
                missing = [n for _, n in slots if n not in present]
                if missing and not self.lenient:
                    raise Refused(
                        f"{rel}: substitution {sub_id}: text not found and"
                        f" {', '.join(missing)} not used in the file (the sentence was"
                        f" rewritten or the mapping is stale; --lenient to continue):"
                        f" {template[:90]!r}"
                    )
                for old, name in slots:
                    state = (
                        "text edited since; macro still used"
                        if name not in missing
                        else "NOT APPLIED: text not found (--lenient)"
                    )
                    if name in missing:
                        self.warn(f"{rel}: substitution {sub_id}: {state}")
                    self.report["subs"].append(
                        {
                            "id": sub_id,
                            "file": rel,
                            "line": None,
                            "old": old,
                            "macro": name,
                            "new": self.macros[name]["printed"],
                            "state": state,
                        }
                    )
                continue
            edits = []
            for i, (old, name) in enumerate(slots):
                a, b = m.span(f"s{i}")
                current = text[a:b]
                state = (
                    "already applied" if current.startswith("\\" + name) else "applied"
                )
                if state == "applied":
                    edits.append((a, b, "\\" + name + "{}"))
                self.report["subs"].append(
                    {
                        "id": sub_id,
                        "file": rel,
                        "line": line_of(text, a),
                        "old": old,
                        "macro": name,
                        "new": self.macros[name]["printed"],
                        "state": state,
                    }
                )
            for a, b, rep in sorted(edits, reverse=True):
                text = text[:a] + rep + text[b:]
            self.put(rel, text)
        # line numbers in the substituted text (shifted by the markers later)
        for entry in self.report["subs"]:
            if entry["line"] is None:
                continue
            sub = next(s for s in SUBS if s[0] == entry["id"])
            m, slots = find_unique(sub[2], self.get(entry["file"]), entry["id"])
            i = [n for _, n in slots].index(entry["macro"])
            entry["line"] = line_of(self.get(entry["file"]), m.start(f"s{i}"))

    def insert_markers(self):
        by_file = {}
        for claim_id, rel, anchor in MARKERS:
            by_file.setdefault(rel, []).append((claim_id, anchor))
        placed = set()
        for rel, specs in by_file.items():
            text = self.get(rel)
            inserts = []
            marked_lines = []  # line numbers (before insertion) that get a marker
            for order, (claim_id, anchor) in enumerate(specs):
                m, _ = find_unique(
                    anchor,
                    text,
                    f"{rel}: REPRO-CHECK anchor for {claim_id}",
                    self.reapply,
                )
                if m is None:
                    self.report["markers_not_placed"].append(
                        {"claim": claim_id, "file": rel, "anchor": anchor}
                    )
                    self.warn(
                        f"{rel}: no REPRO-CHECK marker for {claim_id}: anchor not"
                        f" found (sentence rewritten?): {anchor[:60]!r}"
                    )
                    continue
                start = text.rfind("\n", 0, m.start()) + 1
                indent = re.match(r"[ \t]*", text[start:]).group(0)
                claim = self.claims.get(claim_id)
                if claim is None:
                    state, wording, provisional = (
                        "undetermined (claim missing in numbers.json)",
                        "",
                        False,
                    )
                else:
                    r = claim.get("result")
                    state = {True: "true", False: "false"}.get(r, "undetermined")
                    wording, provisional = claim.get("wording", ""), bool(
                        claim.get("provisional")
                    )
                line = f"{indent}{MARKER_PREFIX}{claim_id} is now {state}" + (
                    " (provisional)" if provisional else ""
                )
                inserts.append((start, order, line, claim_id, state, wording))
                marked_lines.append(line_of(text, start))
                placed.add(claim_id)
            for start, order, line, *_ in sorted(
                inserts, key=lambda x: (x[0], x[1]), reverse=True
            ):
                text = text[:start] + line + "\n" + text[start:]
            self.put(rel, text)
            for entry in self.report["subs"]:
                if entry["file"] == rel and entry["line"] is not None:
                    entry["line"] += sum(1 for n in marked_lines if n <= entry["line"])
            # report with the final line numbers
            lines = text.split("\n")
            for i, ln in enumerate(lines, 1):
                if ln.lstrip().startswith(MARKER_PREFIX):
                    cid = ln.strip()[len(MARKER_PREFIX) :].split(" ")[0]
                    st = ln.strip().split(" is now ", 1)[1]
                    claim = self.claims.get(cid, {})
                    j = i
                    while lines[j - 1].lstrip().startswith(MARKER_PREFIX):
                        j += 1
                    self.report["markers"].append(
                        {
                            "claim": cid,
                            "state": st,
                            "file": rel,
                            "line": i,
                            "sentence_line": j,
                            "sentence": lines[j - 1].strip(),
                            "wording": claim.get("wording", ""),
                            "check": claim.get("check", ""),
                        }
                    )
        self.report["claims_without_marker"] = sorted(set(self.claims) - placed)
        unknown = sorted({c for c, _, _ in MARKERS} - set(self.claims))
        self.report["markers_for_unknown_claims"] = unknown

    # -- checks -------------------------------------------------------------
    def check(self, compiled):
        defined = set(self.macros)
        problems = []
        for rel in self.text:
            if rel.endswith(".tex") and rel not in GENERATED_TEX:
                if brace_balance(self.orig[rel]) != brace_balance(self.text[rel]):
                    problems.append(
                        f"{rel}: brace balance changed {brace_balance(self.orig[rel])} -> {brace_balance(self.text[rel])}"
                    )
        used = set()
        for rel in compiled:
            for name in macro_uses(self.get(rel)):
                used.add(name)
                if name not in defined:
                    problems.append(f"{rel}: \\{name} is not defined in numbers.tex")
        if problems:
            raise Refused("checks failed:\n  " + "\n  ".join(problems))
        self.report["macros_used"] = sorted(used)
        self.report["macros_used_not_final"] = sorted(
            n for n in used if n in self.macros and "?" in self.macros[n]["printed"]
        )

    def scan_numbers(self, compiled):
        """Numbers of the compiled text that are not macros: kept or to review."""
        for rel in compiled:
            if rel == "main.tex":
                continue  # front matter only (title, authors, e-mail addresses)
            text = self.get(rel)
            masked = mask(text).replace("\0", " ")  # comments read as blanks
            prose = NON_PROSE.sub(lambda m: " " * len(m.group(0)), masked)
            spans = []
            for glob_, rx, reason in KEEP:
                if fnmatch.fnmatch(rel, glob_):
                    for m in re.finditer(rx, masked, re.S):
                        spans.append((m.start(), m.end(), reason))
            tokens = [(m.start(), m.end(), m.group(0)) for m in NUMBER.finditer(prose)]
            tokens += [
                (m.start(), m.end(), m.group(0)) for m in WORD_NUMBER.finditer(prose)
            ]
            for a, b, tok in sorted(tokens):
                reason = next((r for s, e, r in spans if s <= a and b <= e), None)
                ctx = re.sub(
                    r"\s+", " ", masked[max(0, a - 50) : b + 40].replace("\0", "")
                ).strip()
                entry = {
                    "file": rel,
                    "line": line_of(text, a),
                    "number": tok,
                    "context": ctx,
                }
                if reason:
                    entry["reason"] = reason
                    self.report["kept_numbers"].append(entry)
                else:
                    experimental = (
                        "." in tok
                        or tok.isalpha()
                        or bool(EXPERIMENTAL_AFTER.match(prose, b))
                    )
                    entry["experimental_looking"] = experimental
                    self.report["review_numbers"].append(entry)

    def scan_plain_text(self):
        """Old numbers in plain-text copies (texts/*.txt): not compiled, update by hand."""
        olds = {}
        for _, _, t in SUBS:
            for old, name in compile_template(t)[1]:
                if re.fullmatch(r"-?\d+\.\d+|\d{3,}", old):
                    olds.setdefault(old, set()).add(name)
        texts = os.path.join(self.paper, "texts")
        for f in sorted(os.listdir(texts)):
            if not f.endswith(".txt"):
                continue
            text = read(os.path.join(texts, f))
            for old, names in sorted(olds.items()):
                for m in re.finditer(
                    r"(?<![\d.])" + re.escape(old) + r"(?![\d])", text
                ):
                    new = ", ".join(
                        f"\\{n} = {self.macros[n]['printed']}" for n in sorted(names)
                    )
                    self.report["plain_text"].append(
                        {
                            "file": f"texts/{f}",
                            "line": line_of(text, m.start()),
                            "old": old,
                            "new": new,
                        }
                    )

    # -- output -------------------------------------------------------------
    def changed_text(self):
        return sorted(r for r in self.text if self.text[r] != self.orig.get(r))

    def changed_binary(self):
        return sorted(r for r, (old, new) in self.binary.items() if old != new)

    def diff(self):
        out = []
        for rel in self.changed_text():
            out += difflib.unified_diff(
                (self.orig.get(rel) or "").splitlines(keepends=True),
                self.text[rel].splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        for rel in self.changed_binary():
            old, new = self.binary[rel]
            o = hashlib.sha256(old).hexdigest()[:12] if old is not None else "none"
            out.append(
                f"Binary file {rel} changes (sha256 {o} -> {hashlib.sha256(new).hexdigest()[:12]})\n"
            )
        return "".join(out)

    def write(self):
        for rel in self.changed_text():
            with open(os.path.join(self.paper, rel), "w") as f:
                f.write(self.text[rel])
        for rel in self.changed_binary():
            with open(os.path.join(self.paper, rel), "wb") as f:
                f.write(self.binary[rel][1])


def run(
    paper,
    out_dir,
    qalm_config=None,
    dry_run=False,
    allow_partial=False,
    refresh_blocks=False,
    lenient=False,
):
    """Compute (and unless dry_run, write) the patch; return (patch, info)."""
    for sub in ("main.tex", "texts", "figtex", "figure"):
        if not os.path.exists(os.path.join(paper, sub)):
            raise Refused(f"--paper {paper} has no {sub}")
    numbers, qalm_config, info = load_inputs(out_dir, qalm_config, allow_partial)
    defined = set(numbers["macros"])
    wanted = {n for _, _, t in SUBS for _, n in compile_template(t)[1]}
    wanted |= {
        n
        for b in REVIEW_BLOCKS
        for n in re.findall(r"\\(num[A-Za-z]+)", block_text(b[0], qalm_config))
    }
    if wanted - defined:
        raise Refused(
            f"macros used by the patch but not in numbers.tex: {sorted(wanted - defined)}"
        )
    p = Patch(paper, out_dir, numbers, qalm_config, refresh_blocks, lenient)
    for msg in info["figure_source_problems"]:
        p.warn(f"figures (--allow-partial): {msg}")
    # A paper that already inputs numbers.tex was patched before: sentences
    # may have been rewritten by hand since, so a missing review block or
    # marker anchor (or a rewritten sentence that still uses its macros) is
    # reported instead of refused.
    p.reapply = bool(
        re.search(r"\\input\{figtex/numbers(?:\.tex)?\}", mask(p.get("main.tex")))
    )
    info["reapply"] = p.reapply
    compiled = p.compiled_files()
    marker_files = sorted(
        {rel for _, rel, _ in MARKERS} | {rel for rel in compiled if rel != "main.tex"}
    )
    p.strip_markers(marker_files)
    p.apply_blocks()
    p.apply_subs()
    p.add_numbers_input()
    p.insert_markers()
    p.copy_outputs()
    p.check(compiled)
    p.scan_numbers(compiled)
    p.scan_plain_text()
    info["dry_run"] = dry_run
    info["changed_text_files"] = p.changed_text()
    info["changed_binary_files"] = p.changed_binary()
    if not dry_run:
        p.write()
    return p, info


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(p, info, out=sys.stdout):
    r = p.report
    w = out.write
    w(
        f"== paper_patch: QALM config {info['qalm_config']}; numbers from {info['numbers_results']}\n"
    )
    if info["provisional_macros"]:
        w(
            f"   WARNING: {len(info['provisional_macros'])} macros are provisional or missing ('?'/'??')\n"
        )
    mode = "would change" if info["dry_run"] else "changed"
    w(
        f"   {mode}: {len(info['changed_text_files'])} text files, {len(info['changed_binary_files'])} binary files\n"
    )
    w("\n== copied outputs\n")
    for c in r["copied"]:
        w(f"   {c['state']:9s} {c['file']}\n")
    w(f"\n== main.tex: {r['main_tex']}\n")
    n_app = sum(s["state"] == "applied" for s in r["subs"])
    w(
        f"\n== substitutions: {n_app} applied, {len(r['subs']) - n_app} already applied\n"
    )
    for s in r["subs"]:
        where = f"{s['file']}:{s['line'] if s['line'] is not None else '?'}"
        w(
            f"   {where:<28s} {s['old'][:28]:>12s} -> \\{s['macro']}{{}} = {s['new']}   [{s['id']}, {s['state']}]\n"
        )
    w(
        "\n== review blocks (check the wording; delete the two marker lines once reviewed)\n"
    )
    for b in r["blocks"]:
        w(f"   {b['file']}:{b['line'] or '?'}  {b['id']}: {b['state']}\n")
    order = {"false": 0, "undetermined": 1}
    marks = sorted(
        r["markers"],
        key=lambda m: (order.get(m["state"].split(" ")[0], 2), m["file"], m["line"]),
    )
    n_false = sum(m["state"].startswith("false") for m in marks)
    n_und = sum(m["state"].startswith("undetermined") for m in marks)
    w(
        f"\n== REPRO-CHECK markers: {len(marks)} ({n_false} false, {n_und} undetermined); rewrite false sentences by hand\n"
    )
    for m in marks:
        w(f"   {m['state']:<13s} {m['file']}:{m['line']:<4d} {m['claim']}\n")
        if not m["state"].startswith("true"):
            w(
                f"        wording: {m['wording'][:150]}\n        check:   {m['check'][:150]}\n"
            )
    for x in r["markers_not_placed"]:
        w(f"   NOT PLACED    {x['file']}  {x['claim']} (anchor not found)\n")
    if r["claims_without_marker"]:
        w(
            f"   claims of numbers.json without a marker: {', '.join(r['claims_without_marker'])}\n"
        )
    if r["macros_used_not_final"]:
        w(
            f"\n== macros used in the paper with provisional/missing values: {', '.join(r['macros_used_not_final'])}\n"
        )
    review = [x for x in r["review_numbers"] if x["experimental_looking"]]
    other = [x for x in r["review_numbers"] if not x["experimental_looking"]]
    w(
        f"\n== numbers left in the compiled text: {len(r['kept_numbers'])} kept on purpose, {len(review)} experimental-looking to review, {len(other)} other\n"
    )
    for x in review:
        w(f"   REVIEW {x['file']}:{x['line']} {x['number']!r}: ...{x['context']}...\n")
    for x in other:
        w(f"   other  {x['file']}:{x['line']} {x['number']!r}: ...{x['context']}...\n")
    if r["plain_text"]:
        w("\n== plain-text copies (not compiled, no macros possible): update by hand\n")
        for x in r["plain_text"]:
            w(f"   {x['file']}:{x['line']} {x['old']} -> {x['new']}\n")
    by_reason = {}
    for x in r["kept_numbers"]:
        by_reason.setdefault(x["reason"], []).append(x)
    for reason, xs in sorted(by_reason.items()):
        locs = ", ".join(
            f"{x['file'].split('/')[-1]}:{x['line']} {x['number']}" for x in xs[:8]
        )
        more = f" (+{len(xs) - 8} more)" if len(xs) > 8 else ""
        w(f"   kept ({len(xs)}): {reason}\n        {locs}{more}\n")


def print_warnings(p, out=sys.stdout):
    if p.report["warnings"]:
        out.write(f"\n== WARNINGS ({len(p.report['warnings'])})\n")
        for msg in p.report["warnings"]:
            out.write(f"   {msg}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", required=True, help="paper checkout to patch in place")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="dir with the generated tables, figures and numbers",
    )
    ap.add_argument("--qalm-config", choices=["a", "b"], default=None)
    ap.add_argument(
        "--dry-run", action="store_true", help="print a unified diff, write nothing"
    )
    ap.add_argument(
        "--allow-partial", action="store_true", help="patch with provisional numbers"
    )
    ap.add_argument(
        "--refresh-blocks",
        action="store_true",
        help="regenerate review blocks edited by hand",
    )
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="on a paper patched before, continue when a substitution's text is gone",
    )
    ap.add_argument("--report", help="also write the report as JSON to this file")
    args = ap.parse_args()
    try:
        p, info = run(
            args.paper,
            args.out_dir,
            args.qalm_config,
            dry_run=args.dry_run,
            allow_partial=args.allow_partial,
            refresh_blocks=args.refresh_blocks,
            lenient=args.lenient,
        )
    except Refused as e:
        print(f"paper_patch: refused, nothing written: {e}", file=sys.stderr)
        return 1
    if args.dry_run:
        sys.stdout.write(p.diff())
        print()
    print_report(p, info)
    print_warnings(p)
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"info": info, **p.report}, f, indent=1, sort_keys=True)
    review = [x for x in p.report["review_numbers"] if x["experimental_looking"]]
    bad = (
        review
        or p.report["claims_without_marker"]
        or p.report["warnings"]
        or p.report["plain_text"]
    )
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
