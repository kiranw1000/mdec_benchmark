# 🏆 4th Monocular Depth Estimation Challenge @ CVPR 2025

<p align="center">
<img src="./assets/syns/image_0026.png" alt="image_0026"  width="250"/>
<img src="./assets/syns/image_0254.png" alt="image_0254"  width="250"/>
<img src="./assets/syns/image_0698.png" alt="image_0698"  width="250"/>
 
<img src="./assets/syns/depth_0026.png" alt="depth_0026"  width="250"/>
<img src="./assets/syns/depth_0254.png" alt="depth_0254"  width="250"/>
<img src="./assets/syns/depth_0698.png" alt="depth_0698"  width="250"/>
</p>


Welcome to the official repository of **MDEC 2025**, the **4th Monocular Depth Estimation Challenge**, held at **CVPR 2025**.

This repository is forked from the original SYNS-Patches starter pack and has been updated with the latest advancements to support this year’s competition.

### ⚡ What’s new in MDEC 2025?

- 📐 **New prediction types**: The challenge became more accessible thanks to the added support of affine-invariant predictions. Metric and scale-invariant predictions are also automatically supported. Disparity predictions, which were supported in previous challenges, are also accepted. 

- 🤗 **Pre-trained Model Support**: We provide ready-to-use scripts for off-the-shelf methods: [Depth Anything V2](mdec_2025/depth_anything_v2/generate.py) (disparity) and [Marigold](mdec_2025/marigold_v1-0/generate.py) (affine-invariant). Adding new pre-trained methods is very easy.

- 📊 **Updated Evaluation Pipeline:** The Codalab grader code has been [updated](src/core/evaluator.py) to accommodate the newly supported prediction types.

### 🚀 More information is available on the official [<img src="https://img.shields.io/badge/%F0%9F%A4%8D%E2%80%83Challenge%20-Website-blue" height=16px alt="Website Badge">](https://jspenmar.github.io/MDEC/)
