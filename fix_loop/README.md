# Fix loop bundle

- `before/`: Kaggle run of the notebook before the fix (commit `d65568f`): digest, validation log and the photo and video run logs. Every photo and video run ended with a RuntimeError and wrote no report.
- `after/`: Kaggle run of the notebook after the fix (commit `57c22eb`, GPU build of COLMAP): digest, validation log, logs, and for every photo and video run the `report.json` and `floor_plan.png`.
- Regenerate: run `notebooks/floorplan_scanner_kaggle.ipynb` on Kaggle (or `tools/run_sample_data.sh photo video` locally) at each commit. Diff: `git diff d65568f 57c22eb`.
- The declaration is in `../declaration.md`.
- `before/real_video/` and `after/real_video/`: the follow-up loop on real walkthrough videos (first video before the scale, alignment and damage fixes; re-recorded landscape video after). Each holds `report.json` and `floor_plan.png`. See the follow-up section of `declaration.md`.
