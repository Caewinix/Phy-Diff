# Phy-Diff
Phy-Diff is a Diffusion Model (DM) based framework for dMRI image synthesis. The algorithm is elaborated in our paper [Phy-Diff: Physics-guided Hourglass Diffusion Model for Diffusion MRI Synthesis](https://arxiv.org/abs/2406.03002).
# Overview
The physics information ADC atlas interacts with the original image at b=n during the forward process. After timesteps diffusing, the image turns into the pure noise. During the reverse process, HDiT predicts the added noise in each timestep in the pixel space, while using the b-vector, b-value as conditions via query-based conditional mapping. After training the noise prediction network, the XTRACT adapter is used to incorporate tractography structure $c_X$. By sampling from the learned pattern, the synthesized image $\hat{S_{n,0}}$ is obtained.
![Schematic diagram](https://github.com/Caewinix/Phy-Diff/blob/main/figures/architecture.png)
# BibTeX
```
@misc{zhang2024phydiffphysicsguidedhourglassdiffusion,
      title={Phy-Diff: Physics-guided Hourglass Diffusion Model for Diffusion MRI Synthesis}, 
      author={Juanhua Zhang and Ruodan Yan and Alessandro Perelli and Xi Chen and Chao Li},
      year={2024},
      eprint={2406.03002},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2406.03002}, 
}
```

