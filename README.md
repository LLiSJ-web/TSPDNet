# TSPDNet

## Traffic Semantic Prototype-Guided Driving-Aware Perception for Traffic Salient Object Detection

<p align="center">
  <b>Official implementation of TSPDNet for Traffic Salient Object Detection (TSOD)</b>
</p>

---

## 📌 Introduction

This repository provides the official implementation of **TSPDNet**, a **Traffic Semantic Prototype-Guided Driving-Aware Perception Network** designed for **Traffic Salient Object Detection (TSOD)**.

> **Note:** The complete **training and testing code will be publicly released once the paper is accepted.**

We will continuously update this repository with pretrained models, training scripts, testing code, evaluation tools, and additional experimental results.

---

## 🚀 News

* **[Coming Soon]** Training code will be released after the paper is accepted.
* **[Coming Soon]** Testing code will be released after the paper is accepted.
* **[Coming Soon]** Pretrained model weights will be provided.
* **[Available]** Saliency maps / prediction results can be downloaded for evaluation.

---

## 🔥 Highlights

* **Traffic Semantic Prototype Interaction**
  Introduces traffic semantic prototypes to strengthen semantic-visual interaction through both implicit and explicit interaction mechanisms, enabling more effective perception of traffic-relevant salient objects.

* **Driving-Aware Spatial Perception**
  Exploits bottom-center spatial priors to model the spatial relationship between salient objects and the ego vehicle, improving the perception of safety-critical objects in driving scenarios.

* **Efficient Multi-Scale Local Modeling**
  Incorporates multi-scale local spatial modeling to capture driving-related contextual cues with lower computational overhead and improved inference efficiency.

* **Accuracy–Efficiency Trade-off**
  TSPDNet achieves competitive detection performance on the **TSOD10K** benchmark while improving inference speed, demonstrating its potential for real-time autonomous driving applications.

---

## 🖼️ Saliency Maps

The predicted saliency maps of **TSPDNet** can be downloaded here:

> **[Download TSPDNet Saliency Maps](https://drive.google.com/file/d/1dvpM9u2wEReGDCcOt_dHZUcA2rkuqqZy/view?usp=drive_link)**

The downloaded prediction maps can be directly used with the evaluation script provided in this repository.

---

## 📊 Evaluation

We provide an evaluation script for quantitatively evaluating the predicted saliency maps.

The evaluation supports commonly used salient object detection metrics, such as:

* **MAE** — Mean Absolute Error
* **F-measure**
* **Weighted F-measure**
* **S-measure**
* **E-measure**

### Quick Evaluation

After preparing the prediction maps and corresponding ground-truth masks, simply run:

```bash
python eval.py
```

A recommended directory structure is:

```text
TSPDNet/
├── datasets/
│   └── GT/
│       ├── xxx.png
│       ├── xxx.png
│       └── ...
│
├── results/
│   └── TSPDNet/
│       ├── xxx.png
│       ├── xxx.png
│       └── ...
│
├── eval.py
└── README.md
```

```bash
python eval.py
```

---

## 🛠️ Installation

### Environment Setup

TSPDNet is implemented with **PyTorch** and follows the environment configuration of [VMamba](https://github.com/MzeroMiko/VMamba).

---

## 📁 Dataset Preparation

We conduct the training and evaluation of TSPDNet on the **TSOD10K** dataset.

Please download **TSOD10K** from its official repository:

**[TSOD10K Dataset](https://github.com/mj129/Tramba)**

## 📦 Model Zoo

| Method  | Backbone    | Pretrained Model | Saliency Maps                                                                                     |
| ------- | ----------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| TSPDNet | [Download](https://drive.google.com/file/d/1Aew9Arfv8OPCxTdaJcHTnpwSkYVkqe3Y/view) | Coming Soon      | [Download](https://drive.google.com/file/d/1dvpM9u2wEReGDCcOt_dHZUcA2rkuqqZy/view?usp=drive_link) |

More checkpoints will be released after the paper is accepted.

## 📢 Code Release

**The full training and testing code of TSPDNet will be publicly released once the paper is accepted.**

The release will include all necessary components for reproducing the experimental results reported in the paper.

⭐ **Please consider starring this repository to follow future updates.**

---

## 📧 Contact

If you have any questions regarding TSPDNet, please feel free to open an issue or contact the authors.

---

## Acknowledgements

We sincerely thank the authors of **Tramba** for their valuable work on Traffic Salient Object Detection and for providing the **TSOD10K** dataset and related resources that support this research.
