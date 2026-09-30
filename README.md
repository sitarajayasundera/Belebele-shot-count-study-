# When More Isn't Better: Positional Bias as a Confound in Cross-Lingual Few-Shot Prompting

Empirical study on the effect of shot count in few-shot prompting, comparing English (high-resource) and Sinhala (low-resource) on the Belebele benchmark.

**Poster & appendix:** see the submitted poster PDF and appendix (references + academic integrity declaration) for the full write-up. This repository contains the code and raw results behind it.

## Research question

Does the shot count needed before few-shot accuracy plateaus differ between a high-resource language (English) and a low-resource one (Sinhala)?

## Headline finding

The expected rise-then-plateau pattern did not hold. Instead, at k=8 the primary model (mT0-XL) showed a measurable lean toward a single answer letter ("C") well above the 25% chance rate expected on this 4-way multiple-choice task. A fully independent re-test on data never used in any earlier run confirmed this for English (40.4%, binomial p < .001, holds under a stricter item-level test) but found only marginal support for Sinhala (34.3%, item-level p = .06). A second, architecturally different model (Qwen2.5-7B-Instruct) did not reproduce the English effect but did show a separate, steady decline in Sinhala — the exact bias mechanism differs by model, but the low-resource language was the more vulnerable case in both.

## Data

[Belebele](https://huggingface.co/datasets/facebook/belebele) (Bandarkar et al., 2024), loaded programmatically via the Hugging Face `datasets` library (`facebook/belebele`, configs `eng_Latn` and `sin_Sinh`). No local data files are required — every notebook downloads the dataset on first run.

## Notebooks, in the order they were run

| File | Model | Status | What it contains |
|---|---|---|---|
| `mT0_XL60.ipynb` | mT0-XL | **Main experiment** | The core shot-count sweep (k=0,1,2,4,8; n=60/language), the truncation-bug discovery and fix, qualitative error analysis, the W/X/Y/Z relabeling experiment, and the k=5–7 threshold investigation. |
| `mT0_XL150.ipynb` | mT0-XL | **Superseded** | An early attempt to confirm the k=8 finding at larger scale (n=150). Later found to partially overlap with data already scored in the main notebook (same fixed shuffle seed + deterministic decoding), which inflated the apparent effect size. Kept for transparency; **not** the source of the numbers reported on the poster. |
| `belebele_clean_positional_bias.ipynb` | mT0-XL | **Authoritative confirmation** | Re-tests the k=8 finding on a disjoint slice of the dataset untouched by any earlier run, with full per-item logging and two significance tests (binomial and a stricter per-item paired test). **This is the source of the corrected numbers (40.4% / 34.3%) reported on the poster.** |
| `Qwen-2.ipynb` | Qwen2.5-7B-Instruct | **Generalization check** | Repeats the shot-count sweep on a second, causal-LM architecture to test whether the mT0-XL finding generalizes. Includes a chat-template formatting fix (Qwen requires structured prompts, unlike mT0-XL) and a follow-up Sinhala k=16 run. |

## How to run

1. Open a notebook in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run all cells in order.

**Note on results persistence:** `Qwen-2.ipynb` and `belebele_clean_positional_bias.ipynb` write their output CSVs to a Google Drive path (`/content/drive/MyDrive/belebele_results/...`). Add a `drive.mount('/content/drive')` cell before running if you want results to persist beyond the Colab session; otherwise redirect `output_dir` in the relevant config dictionary to a local path (e.g. `"results"`).

**Note on `mT0_XL60.ipynb`:** its early cells (model loading, GPU diagnostics) reflect a real debugging sequence from a session where the model initially failed to attach to the GPU correctly. A fresh run may not reproduce that original failure; the working configuration is the one used from that point onward in the notebook.

## Results files

Each notebook saves its raw and/or aggregated results as CSVs alongside the code (or to the Drive path noted above). The most important file for verifying the poster's headline claim is `clean_per_item.csv` and `clean_summary.csv`, produced by `belebele_clean_positional_bias.ipynb` — every individual prediction is logged with its row index, seed, raw model output, and correctness, so the reported statistics can be recomputed independently.
