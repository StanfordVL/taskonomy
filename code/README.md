# TASKONOMY Code

This folder contains the code used in the paper. The version of TensorFlow (0.12) is now very outdated, but we are still including the original code here for reference purposes.

**The code is structured as follows**
```python
experiments/  # Example configs (actual configs generated in tools/scripts/)
    transfers/             # Example transfer config
    aws_second/            # Task-specific configs used for paper
    final/                 # Configs used for taskbank (slightly improved)
lib/          # The bulk of the TF code
    data/                  # Dataloading code
    losses/                # Different types of losses (e.g. for GANs)
    models/                # Architectures
    optimizers/            # Ops and train_steps. E.g. for GANs
    savers/                # Code for saving checkpoints to S3
notebooks/    # Jupyter notebooks used for developing
    *.ipynb   # Debugging notebooks, hyperparameter sweeps, etc
    analysis/              # Notebooks that analyze aggregated transfers. Early implementation of BIP solver
    transfer_viz/          # Visualizing results of networks with different HPs
    quality_control/       # Visualizing task-specific network outputs
        bottleneck/        # Results with different bottleneck sizes
    quality_control_final  # Visualizing taskbank networks
tools/      # Utilities
    scripts/               # Scripts to generate configs
    extract_losses.py      # Compute losses on a train/val/test set
    train.py               # Training script
    transfer.py            # Transfer script
```

#### To create the taskonomy, we used the following procedure:

1) Create all task-specific configs (using `tools/scripts/` and saving to `experiments/`)
2) Select which config to run, then train the task-specific network using `tools/train.py`
3) Create all transfer configs (using `tools/scripts/` and saving to `experiments/`)
4) Select which config to run, then train the transfer network using `tools/transfer.py`
5) Compute losses with `extract_losses.py`
6) Generate win rates and affinities with one of the methods in `analysis/`


**Note**: this folder provides the full code and additional resources for archival and information purposes only. We dont maintain the code here. For trained TASK BANK networks and demo code for running them, please see the [TASK BANK folder](https://github.com/StanfordVL/taskonomy/tree/master/taskbank). For Taskonomy dataset, please see the [DATASET folder](https://github.com/StanfordVL/taskonomy/tree/master/data). For more details and the full methodology, please see the [main paper and website](http://taskonomy.vision).

## Citation
If you find the code, models, or data useful, please cite this paper:
```
@inproceedings{zamir2018taskonomy,
  title={Taskonomy: Disentangling Task Transfer Learning},
  author={Zamir, Amir R and Sax, Alexander and and Shen, William B and Guibas, Leonidas and Malik, Jitendra and Savarese, Silvio},
  booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
  organization={IEEE}
}
```
