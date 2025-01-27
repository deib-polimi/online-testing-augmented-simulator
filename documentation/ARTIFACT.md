# Autonomous Driving System Testing with Domain Augmentation

This repository addresses the limited operational design domain (ODD) coverage in current autonomous driving simulators by integrating generative artificial intelligence techniques with physics-based simulation. Specifically, we implement three diffusion-based strategies—Instruction-editing, Inpainting, and Inpainting with Refinement—to produce augmented images representing new ODDs. An automated segmentation-based validator ensures semantic realism and achieved as low as 3% false positives in our human study, preserving image quality and correctness. System-level experiments showed that these domain augmentation methods increase ODD coverage, effectively uncovering previously undetected ADS failures before real-world testing.

<p align="center">
  <img src="udacity-day-to-sunset.gif" width="53.96%" alt="Udacity Day-to-Sunset Preview"/>
  <img src="carla-day-to-night.gif" width="45%" alt="CARLA Day-to-Night Preview"/>
</p>


## Purpose
This artifact provides **domain augmentation** techniques for **Autonomous Driving System (ADS) testing**, using three diffusion-based methods (Instruction-editing, Inpainting, and Inpainting with Refinement). These methods generate augmented driving images of diverse operational design domain (ODD) conditions (e.g., weather, time of day) while preserving the road structure. The artifact facilitates **system-level ADS testing** in both Udacity and CARLA simulators, allowing researchers to uncover failures that standard simulator setups might miss.

### ICSE Artifact Track Badges 

We are submitting this artifact for three ACM Artifact Badges: **Available**, **Functional**, and **Reusable**. We believe our artifact merits the **Available** badge because the code, documentation, and data are deposited in a publicly accessible and citable repository, ensuring long-term accessibility. We claim the **Functional** badge as the artifact is rigorously documented, consistent with the paper, and includes scripts and instructions that allow users to systematically reproduce our main findings. Finally, we contend that the artifact also qualifies for the **Reusable** badge by virtue of its well-structured codebase, detailed environment setup guides, and clearly defined extension points—facilitating future research and repurposing beyond the original study.

## Provenance
This artifact is publicly available in the following online repository:

- **GitHub Repository**: [https://github.com/deib-polimi/online-testing-augmented-simulator](https://github.com/deib-polimi/online-testing-augmented-simulator)

The **preprint** of the paper is available here:

- **Paper Preprint**: [Arxiv](https://arxiv.org/abs/2409.13661)
