### Introduction

MANO is a human hand model which represent human hand as 48D(here we only take into account 45D, without palm) pos. In this work, we reduce the dimension of the MANO pose dataset, using VAE encoder to represent the 45D hand joints representation into 9D latent, then reconstruct the 9D latent back to 45D hand joints.

## Dataset

This project uses MANO pose data processed from the GRAB dataset.

Due to GitHub file size limits, the dataset is hosted externally.

### Download

Download the dataset from Google Drive:
[https://drive.google.com/file/d/XXXX/view?usp=sharing](https://drive.google.com/file/d/12Ky9J66Qd0swwXPEkKNJtF0cfL27qY6A/view?usp=share_link)

After downloading, create a file named "hand_dataset" parellal to directory "Compress", extract the zip file into "hand_dataset"

### Extract data

We use GRAB data set to complete this object. To derive data of MANO Pose from GRAB dataset, we need to prepare our dataset in the first place.
run "data_extract.py", you'll get the npz file: "grab_mano_theta_all_norm.npz" which contains millions of data from MANO pose (shape: N*45).
<img width="172" height="106" alt="e5854559d23b83b36027a82d97ab14c0" src="https://github.com/user-attachments/assets/9f4b06c4-abc4-47f3-9923-7d2272ad2a7d" />
<img width="172" height="106" alt="e5854559d23b83b36027a82d97ab14c0" src="https://github.com/user-attachments/assets/68640865-c573-4c52-bfb9-bc9806f1f0c2" />
<img width="190" height="116" alt="4a72467e2bb906e849cc37cf682d0c1a" src="https://github.com/user-attachments/assets/4babfadc-4d4c-45ae-a9a4-addde070372b" />
<img width="118" height="132" alt="9e3777efcc4476824cead5a370a03c4d" src="https://github.com/user-attachments/assets/251c059b-cd5d-434d-9692-9756d69e78b8" />

### Train VAE

VAE structure can be seen in file "vae.py", and for VAE loss, it's in file "loss.py". You can design your own VAE by modifying these two files.
To train the VAE, run "train_vae.py", this will save a model called "hand_pose_vae_all_z9.pt". By default the latent of VAE is equal to 9.

### Validate your model

run "eval_recon.py", then visualized evaluation matrices will shown in the new file.

Example:
<img width="1200" height="300" alt="per_dim_mse" src="https://github.com/user-attachments/assets/3ec7bc33-e369-434e-b8fb-2440a00147b9" />
<img width="1000" height="300" alt="theta_overlay_04" src="https://github.com/user-attachments/assets/2529f60c-d644-42f2-854b-bc108b401ee4" />
