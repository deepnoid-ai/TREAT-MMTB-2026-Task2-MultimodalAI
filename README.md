# MultimodalAI — TREAT-MMTB 2026 Task 2

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Deepnoid/TREAT-MMTB-2026-Task2-MultimodalAI) [![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/deepnoid-ai/TREAT-MMTB-2026-Task2-MultimodalAI) [![Leaderboard](https://img.shields.io/badge/Leaderboard-Rank%202-2EA44F)](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026/blob/main/leader_board_point.json) [![TREAT-MMTB 2026](https://img.shields.io/badge/MICCAI%202026-TREAT--MMTB-1F6FEB)](https://treat-mmtb.mi2rl.co/)

TB/Normal classification from chest X-ray PNGs for [TREAT-MMTB 2026](https://treat-mmtb.mi2rl.co/). Clinical metadata are not used.

## Challenge result

MultimodalAI's [official final external leaderboard](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026/blob/main/leader_board_point.json) result:

| Rank | External fullset F1 |
| --- | --- |
| 2 | 0.8642 |

## Installation

Python 3.10, with PyTorch for CUDA 11.8 and dependencies from [requirements.txt](requirements.txt):

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## AutoModel inference

Code and weights load automatically from Hugging Face.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Deepnoid/TREAT-MMTB-2026-Task2-MultimodalAI",
    trust_remote_code=True,
).to("cuda").eval()

result = model.predict("/path/to/image.png", output_dir=None)[0]
label = result["label"]

results = model.predict("/path/to/png_folder", output_dir="output/task2", batch_size=4)
```

- **Input:** a PNG file or a folder, read non-recursively with case-insensitive extensions.
- **Return:** always a list of results containing `filename` and `label` (`TB` or `Normal`). `output_dir=None` (default) writes no output.
- **Save:** setting `output_dir` writes `prediction.csv` with columns `filename,TB/Normal`.

Use `.to("cpu")` for CPU inference. The default batch size is 32; reduce it for smaller GPUs.

## CLI inference

Place all four checkpoints in `weights/` beside `predict.py` (or set `--weights`). Architecture and threshold settings are read from `config.json` (or set `--config`). `--input` accepts a file or folder.

```bash
python predict.py --input /path/to/png_folder --output output/task2 --batch-size 4
```

## Models and method

`weights/model_0.safetensors` through `weights/model_3.safetensors` are DINOv3 ViT-L/16 classifiers with attention pooling.

Input combines histogram equalization, CLAHE, and grayscale channels, center-padded and resized to 512 × 512. Inference ensembles four models with horizontal-flip augmentation and a decision threshold of **0.35** for `TB` versus `Normal`.

## Docker

Requires an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Place all four [checkpoints](https://huggingface.co/Deepnoid/TREAT-MMTB-2026-Task2-MultimodalAI/tree/main/weights) in `weights/` beside the Dockerfile. Build with network access; inference runs offline.

```bash
docker build -t multimodalai-task2:latest .
mkdir -p output
docker run --rm --gpus all --network none \
  -v /absolute/path/to/png_folder:/input:ro \
  -v "$PWD/output":/output multimodalai-task2:latest
```

Append `--input /input/image.png` for a single file or `--batch-size 4` for a smaller batch. Results are saved in `output/prediction.csv`.

## Builds on

Vision-language pretraining followed [GLINT](https://arxiv.org/abs/2606.03180) (Park et al., 2026), using Meta AI / FAIR's [DINOv3](https://arxiv.org/abs/2508.10104) encoders (Siméoni et al., 2025), [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) sentence embeddings (Song et al., 2020), and report labels using [Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b).

| Dataset | Use |
| --- | --- |
| [TREAT-MMTB 2026](https://doi.org/10.5281/zenodo.19732124) | Classification training |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | Vision-language pretraining and classification training |
| [TB Portals](https://tbportals.niaid.nih.gov/) | Classification training |
| [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) | Classification training |
| [TBX11K](https://mmcheng.net/tb/) | Classification training |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | Classification training |
| [Shenzhen and Montgomery](https://lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9356.pdf) | Classification training (two of four ensemble members) |

## Acknowledgements

This work was supported by the Technology Innovation Program (RS-2025-02221011, Development of Medical-Specialized Multimodal Hyperscale Generative AI Technology for Global Integration) funded by the Ministry of Trade Industry & Energy (MOTIE, South Korea), and by the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT).

Data were obtained from the [TB Portals](https://tbportals.niaid.nih.gov), which is an open-access TB data resource supported by the National Institute of Allergy and Infectious Diseases (NIAID) Office of Cyber Infrastructure and Computational Biology (OCICB) in Bethesda, MD. These data were collected and submitted by members of the [TB Portals Consortium](https://tbportals.niaid.nih.gov/Partners). Investigators and other data contributors that originally submitted the data to the TB Portals did not participate in the design or analysis of this study (Rosenthal et al., 2017).
