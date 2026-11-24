Pheromone-Inspired Reward Shaping
Return Calibration, Representation, and Memory Retention

The English paper has 7 main-text pages and 1 reference page, with four
Python-generated vector figures. Author details are intentionally blank.
The final PDF is output/pdf/pheromone_shaping_report.pdf relative to the
workspace root. The LaTeX master is paper_report_20261120/main.tex.

What was added for this paper
----------------------------
600 fresh training runs: 20 seeds x 3 navigation conditions x 10 methods;
18 million real interactions. The source, protocol and method were frozen
before inspecting these outcomes. The six primary comparisons validate
improvements over raw concentration in deterministic U-detour and four rooms;
the stochastic advantage does not replicate. State-only score controls are
already competitive, limiting attribution to return calibration or mixing.
A separate exact 5x7 tempting-reward diagnostic checks goal preservation; it
is model planning, not a sampled training or sample-efficiency experiment.

Regenerate every paper figure and table
--------------------------------------
Python 3.13 was used. Install requirements.txt in a suitable environment.
From the unzipped workspace root:
  python paper_report_20261120/make_figures.py

This command needs only the local paper_data.json snapshot, NumPy and
Matplotlib. No connected account, external data download or legacy absolute
path is required. The snapshot contains all 600 fresh final/checkpoint
records, all 400 MountainCar records used by the figure, the aggregate prior
statistics and the exact diagnostic. data_provenance.json lists SHA256 hashes
of the source records. Figures are exported as PDF and high-resolution PNG.

Compile
-------
From paper_report_20261120:
  tectonic -X compile --untrusted main.tex --outdir build
or use a standard LaTeX installation:
  latexmk -pdf -outdir=build main.tex

The bibliography is contained in main.tex; BibTeX is unnecessary. Standard
packages include lmodern, amsmath, natbib, booktabs, placeins and balance.
The first Tectonic build may need network access to cache standard packages.

Recompute the fresh statistics and exact diagnostic
--------------------------------------------------
From the workspace root:
  python paper_report_20261120/analyze_validation.py
  python paper_report_20261120/experiments/goal_diagnostic.py

These use the supplied compatible author code and course environment. The
raw fresh runs and final models are in experiments/fresh_nav/runs and models.
Bootstrap and random sign-flip procedures each use 20000 simulations. The
training seed is the inferential unit. The six primary, fifteen secondary,
and six strength comparisons are separate declared families.

Rerun training without changing frozen outcomes
----------------------------------------------
  python paper_report_20261120/experiments/engine.py confirm --seeds 20 --start 40000 --workers 8 --out validation_repeat --scenarios u_detour,four_rooms,slippery_rooms --methods plain,raw_static,raw_dynamic,state_raw_static,state_raw_matched,return_static,return_gate_unbounded,mc_dynamic_all_unbounded,gate_scale003,gate_scale030

Use a NEW --out name. Existing results cannot be overwritten by the driver.
analyze_validation.py intentionally analyzes the frozen fresh_nav folder;
adapt its input path explicitly if analyzing a separately named rerun.
engine.diff and provenance.json document the exact extension of the frozen
earlier study. The old 3660-run study has not been edited or replaced.

Sources and scope
-----------------
The accompanying archive includes the fresh experiment, plotting snapshot,
minimal source dependencies, and original attribution/license files.
The full earlier 3660-run data/models remain in research_story_20261120.zip;
this compact paper package does not duplicate that older complete archive.
The old source-audit/confirmation summaries used in the paper are included
inside paper_data.json, along with all checkpoint records needed by its plots.
The source paper is Bromuri (2019), DOI 10.1007/s10732-019-09408-x; the released
repository is pinned to 95786b39b9ee8c5e004e71ccfa790304cbb3f684.
Our priors and models are locally generated, not official released artifacts.
References distinguish direct implementation reuse, theory, transferred
heuristic controls, and our additions. AI-assisted coding and drafting are
disclosed in the paper. No team identities or individual contributions were
invented.

Audit records
-------------
validation_summary.json: all fresh means, SDs and declared test families.
discounted_checks.json: supplementary original-discounted-return checks.
audit.json: completion, frozen source, model finiteness and PDF checks.
compile_result.json: actual compiler result.
results_export.csv: all fresh aggregate metrics (UTF-8 with BOM).
FILE_MANIFEST.json: source/data/artifact hashes for the delivery.
