# JPMAP with Hierarchical VAE

Adaptation of the code of [JPMAP github](https://github.com/mago876/JPMAP) of the article of:
[González, M., Almansa, A., & Tan, P. (2021). Solving Inverse Problems by Joint Posterior Maximization with Autoencoding Prior. arXiv preprint arXiv:2103.01648.](https://arxiv.org/abs/2103.01648)  

School project, attempt to change the used VAE in JPMAP by HDN (Hierarchical DivNoising) designed in [Prakash, Mangal / Delbracio, Mauricio / Milanfar, Peyman / Jug, Florian 
Interpretable Unsupervised Diversity Denoising and Artefact Removal 
2022 ](https://arxiv.org/pdf/2104.01374.pdf)
Some of the code of [this paper's repo](https://github.com/juglab/HDN) is used here.

To install the needed conda environment, run `conda env create -f jpmap_env.yml`.  
Then activate the environment with `conda activate jpmap`.  

The bash script `run_experiments.sh` contains example code for running JPMAP for different inverse problems.  You need 
to have a pretrained HDN network stored in `pretrained_models/model_name/model.net`.

The parameters for running JPMAP are described in `run_algorithms_hdn.py`.
