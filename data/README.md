# Data

This project uses the **WESAD (Wearable Stress and Affect Detection)** dataset
(Schmidt et al., 2018).

## Download
1. Go to the WESAD dataset home page (University of Siegen), linked from
   https://archive.ics.uci.edu/dataset/465/wesad
2. Download `WESAD.zip` (~2 GB) and unzip it.
3. Place the subject folders here so the structure is:

    data/WESAD/S2/S2.pkl
    data/WESAD/S3/S3.pkl
    ...
    data/WESAD/S17/S17.pkl

## Notes
- 15 subjects: S2–S17. S1 and S12 were excluded by the dataset authors
  due to sensor malfunction.
- Raw data is **not included** in this repository because of its size and
  license. Please download it from the official source.
- Processed windows (`.npz`) are generated into `data/processed/` by the
  preprocessing script and are also not tracked.

## Citation
Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
Introducing WESAD, a multimodal dataset for wearable stress and affect detection.
*Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI)*.