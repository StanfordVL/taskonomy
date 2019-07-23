# [Taskonomy: Disentangling Task Transfer Learning](https://taskonomy.vision/)

This folder contains the **raw and post-processed measurements of task affinities**, under various configurations (e.g. 1k/16k/etc images used for training transfer functions and measuring task affinities). The **win-rates** are included as well. 

This data was used for generating the task affinities reported in the figures 7 and 13 of the [paper](http://taskonomy.stanford.edu/#paper) (See below) and the transfer learning analysis. For more details, please see section 3.3 "Step III: Ordinal Normalization using Analytic
Hierarchy Process (AHP)" of the [paper](http://taskonomy.stanford.edu/#paper).
 
<div align="center">
  <img src="https://github.com/StanfordVL/taskonomy/raw/master/data/assets/affinity_pre_post_AHP.jpg"  width="900px" />
</div>

**First-order task affinity matrix before (left) and after (right)
Analytic Hierarchy Process (AHP) normalization.** Lower means better
transferred. (figure 7 of the [paper](http://taskonomy.stanford.edu/#paper) )



<div align="center">
  <img src="https://github.com/StanfordVL/taskonomy/raw/master/data/assets/task_tree.jpg"  width="600px" />
</div>

**Task Similarity Tree.** Agglomerative clustering of tasks
based on their transferring-out patterns (i.e. using columns of normalized
affinity matrix as task features). 3D, 2D, low dimensional geometric, and
semantic tasks clustered together using a fully computational approach. (figure 14 of the [paper](http://taskonomy.stanford.edu/#paper) )
 

 


## Citation
If you find the affinities, code, models, or data useful, please cite this paper:
```
@inproceedings{zamir2018taskonomy,
  title={Taskonomy: Disentangling Task Transfer Learning},
  author={Zamir, Amir R and Sax, Alexander and and Shen, William B and Guibas, Leonidas and Malik, Jitendra and Savarese, Silvio},
  booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
  organization={IEEE}
}
 
