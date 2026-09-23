# AMOC_Detector

I worked on this during my time at Lancaster University on the STOR-i programme, looking at changepoint detection.

## Key work

| File | What it contains |
|---|---|
| [01 Intro Work](01_Intro_Work.ipynb) | Learning notebook: PELT and penalty choice, parametric (L2) vs kernel (RBF) costs, multiple changepoints, segmentation metrics, and the first version of the AMOC test |
| [Notes](Notes.md) | Theory notes: MLE, the kernel trick, characteristic kernels, and evaluation metrics (F1, Rand index, Hausdorff distance, power curves) |
| [02 Statistical Testing](02_Statistical_Testing.ipynb) | Monte Carlo power analysis of the AMOC detector across mean, variance, skewness, kurtosis and multimodal changes |

The detector itself lives in [`src/AMOC_Detector`](src/AMOC_Detector), with worked examples in [`examples`](examples). Power results are saved in [`data`](data) and figures in [`plots`](plots).
