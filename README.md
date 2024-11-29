# Autonomous Driving System Testing with Domain Augmentation

[//]: # (<p align="center">)

[//]: # (  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="Recommender System 2018 Challenge Polimi" />)

[//]: # (</p>)

## Requirements
To set up and run this project, ensure you have the following:

- **Python 3.8+**: The project is built on Python, so make sure you have the correct version installed. Experiments were executed with Python3.10
- **CUDA-compatible GPU**: Essential for running the deep learning models efficiently. Experiments were executed with CUDA 12.1
- **Required Python packages**: Install the dependencies using the following command:

  ```bash
  pip install -r requirements.txt
  ```
  
## Udacity Simulator
We utilized the Udacity Self-Driving Car Simulator as the simulation environment for this project. 
The simulator supports various weather conditions, times of day, and road types, making it ideal for testing Autonomous Driving Systems (ADS) under different Operational Design Domains (ODDs).
The experiments were run on Linux. 

- Download the simulator from this [link](https://icse-2025.s3.eu-north-1.amazonaws.com/udacity-linux.tar.xz).

## Augmented Datasets

The dataset used in this project consists of image pairs generated from multiple ODD domains. 

We augmented the images collected from the Udacity simulator using three domain augmentation techniques and applied them to create new training and testing scenarios.

### Instruction-editing
<p align="center">
  <img src="images/instruction-editing/change-season-to-autumn_0.jpg" width="19%"/>
  <img src="images/instruction-editing/change-season-to-autumn_1.jpg" width="19%"/> 
  <img src="images/instruction-editing/change-season-to-autumn_2.jpg" width="19%"/>
  <img src="images/instruction-editing/change-season-to-autumn_3.jpg" width="19%"/> 
  <img src="images/instruction-editing/change-season-to-autumn_4.jpg" width="19%"/>
</p>

The augmented dataset can be accessed from this [link](https://icse-2025.s3.eu-north-1.amazonaws.com/instructpix2pix.tar.xz)

### Inpainting
<p align="center">
  <img src="images/inpainting/A-street-in-autumn-season-photo-taken-from-a-car_0.jpg" width="19%"/>
  <img src="images/inpainting/A-street-in-autumn-season-photo-taken-from-a-car_1.jpg" width="19%"/>
  <img src="images/inpainting/A-street-in-autumn-season-photo-taken-from-a-car_2.jpg" width="19%"/>
  <img src="images/inpainting/A-street-in-autumn-season-photo-taken-from-a-car_3.jpg" width="19%"/>
  <img src="images/inpainting/A-street-in-autumn-season-photo-taken-from-a-car_4.jpg" width="19%"/>
</p>

The augmented dataset can be accessed from this [link](https://icse-2025.s3.eu-north-1.amazonaws.com/stable_diffusion_inpainting.tar.xz)

### Inpainting with Refinement
<p align="center">
  <img src="images/inpainting-with-refinement/a-street-in-autumn-season_0.jpg" width="19%"/>
  <img src="images/inpainting-with-refinement/a-street-in-autumn-season_1.jpg" width="19%"/>
  <img src="images/inpainting-with-refinement/a-street-in-autumn-season_2.jpg" width="19%"/>
  <img src="images/inpainting-with-refinement/a-street-in-autumn-season_3.jpg" width="19%"/>
  <img src="images/inpainting-with-refinement/a-street-in-autumn-season_4.jpg" width="19%"/>
</p>

The augmented dataset can be accessed from this [link](https://icse-2025.s3.eu-north-1.amazonaws.com/stable_diffusion_inpainting_controlnet_refining.tar.xz)

## Human Study

The human study and the documentation about its results can be found [here](documentation/human_study.md)

<p align="center">
  <img src="documentation/realism_example.png" width="45%" alt="Realism Question Screenshot" />
  <img src="documentation/semantic_example.png" height="45%" alt="Semantic Validity Question Screenshot" />
</p>
