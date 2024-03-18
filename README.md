# Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing
The official repo for "[Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](https://arxiv.org/pdf/2403.10050.pdf)"

<p align="center">
<a href="https://arxiv.org/pdf/2403.10050.pdf"><img src="https://img.shields.io/badge/Arxiv-2403.10050-B31B1B.svg"></a>
<a href="https://slothfulxtx.github.io/TexGS/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
</p>

## :mega: Updates

[18/3/2024] The project page is created.

[18/3/2024] The official repo is initialized.

## Abstract

3D Gaussian splatting, emerging as a groundbreaking approach, has drawn increasing attention for its capabilities of high-fidelity reconstruction and real-time rendering. However, it couples the appearance and geometry of the scene within the Gaussian attributes, which hinders the flexibility of editing operations, such as texture swapping. To address this issue, we propose a novel approach, namely Texture-GS, to disentangle the appearance from the geometry by representing it as a 2D texture mapped onto the 3D surface, thereby facilitating appearance editing. Technically, the disentanglement is achieved by our proposed texture mapping module, which consists of a UV mapping MLP to learn the UV coordinates for the 3D Gaussian centers, a local Taylor expansion of the MLP to efficiently approximate the UV coordinates for the ray-Gaussian intersections, and a learnable texture to capture the fine-grained appearance. Extensive experiments on the DTU dataset demonstrate that our method not only facilitates high-fidelity appearance editing but also achieves real-time rendering on consumer-level devices, e.g. a single RTX 2080 Ti GPU.

## Texture Swapping with Texture-GS

<p align="center">
<img src="assets/teaser.png" width="800"/>
</p>

## TODO

- [x] Release the demo page and more video results.
- [ ] Release the code. 

## Citation

If you find this code useful for your research, please consider citing:
```
@misc{xu2024texturegs,
    title={Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing}, 
    author={Tian-Xing Xu and Wenbo Hu and Yu-Kun Lai and Ying Shan and Song-Hai Zhang},
    year={2024},
    eprint={2403.10050},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements

This project is built on source codes shared by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).

