# Actor-Critic Instance Segmentation
Official implementation of the actor-critic model (Lua Torch) to accompany our paper

> Nikita Araslanov, Constantin Rothkopf and Stefan Roth, **Actor-Critic Instance Segmentation**, CVPR 2019.

- ArXiv preprint: [https://arxiv.org/abs/1904.05126](https://arxiv.org/abs/1904.05126)
- CVPR 2019 proceedings: [PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Araslanov_Actor-Critic_Instance_Segmentation_CVPR_2019_paper.pdf)
- Contact: Nikita Araslanov, [fname.lname]@visinf.tu-darmstadt.de

## Installation
We tested the code with ```Lua 5.2.4```, ```CUDA 8.0``` and ```cuDNN-5.1``` on Ubuntu 16.04.
- To install Lua Torch, please, follow the [official docs](http://torch.ch/docs/getting-started.html#_)
- CuDNN 5.1 and CUDA 8.0 are available on the the [nvidia website](https://developer.nvidia.com)
- Compile the Hungarian algorithm implementation (```hungarian.c```):
```
cd <acis>/code && make
```

## Running the code
### Training
Training the actor-critic model from scratch consists of four steps:
1) Training the pre-processing network to predict angle quantisation
2) Generating augmented data with the official data from the pre-processing network
3) Pre-training the decoder part of the actor-critic model
4) Training actor-critic on sequential prediction length

These steps are described in detail for CVPPP dataset next. You can download the dataset from the [offical website](https://www.plant-phenotyping.org/datasets-download). For instance segmentation, we use only A1 subset of the dataset, 128 annotated images in total.

The commands referenced below assume ```<ACIS>/code``` as the current directory, unless mentioned otherwise.

#### Training pre-processing network
As described in our paper, the actor-critic model uses angle quantisation [1] and the foreground mask, following Ren & Zemel [2].
The pre-processing network is an FCN and can be trained using ```main_preproc.lua```.
A bash script ```runs/cvppp_preproc.sh``` contains an example command for training the network.
Running
```
./runs/cvppp_preproc.sh
```
will create a directory ```checkpoints/cvppp_preproc```, where intermediate checkpoints, validation results and the training progress will be logged.

> [1] Uhrig J., Cordts M., Franke U., and Brox T. Pixel-level encoding and depth layering for instance-level semantic
labeling. In GCPR, 2016.<br>
> [2] Ren M. and Zemel R. End-to-end instance segmentation and counting with recurrent attention. In CVPR, 2017.

#### Generating augmented data
Instead of keeping the pre-processing net around while training the actor-critic model, we will generate the axiliary data (*with augmentation*) using the pre-processing net: the angles and the foreground. This will save GPU memory and improve runtime at the expense of some disk space.
```
./runs/cvppp_preproc_save.sh
```
The script will iterate through the dataset (300 epochs for CVPPP), each time with random augmentation switched on, such as rotation and flipping. You can change the amount of data generated using parameter ```-preproc_epoch``` (see ```cvppp_preproc_save.sh```). 
By default, you should account for 80GB of generated data (disk space). The data will be saved into ```data/cvppp/A1_RAW/train/augm``` and ```data/cvppp/A1_RAW/val/augm```. For the next steps, please, move the augmented data into ```data/cvppp/A1_AUG/train``` and ```data/cvppp/A1_AUG/val/```:
```
mv data/cvppp/A1_RAW/train/augm/* data/cvppp/A1_AUG/train/
mv data/cvppp/A1_RAW/val/augm/* data/cvppp/A1_AUG/val/
```

#### Pre-training
The purpose of the pre-training stage is to learn a compact representation for masks (action space). The training is equivalent to Variational Auto-Encoder (VAE), where the reconstruction loss is computed for one target mask.
Assuming the augmented data generated in the previous step, run
```
./runs/cvppp_pretrain.sh
```
The script will use ```pretrain_main.lua``` to create and train the actor model. It will also reduce the learning rate in stages. After the training, the script will create the checkpoints for the decoder, and the encoder trained to predict one mask, both used in the final training step, described next.

#### Training
```main.lua``` is the entry script to train the actor-critic model on sequential prediction. The logging is realised with [crayon](https://github.com/torrvision/crayon). Please, follow [this README](https://github.com/arnike/acis_release/blob/master_release/code/README.md) to set it up.

To train *BL-Trunc* (baseline with truncated backprop), run
```
./runs/cvppp_train_btrunc.sh
```
To train *AC* (the actor-critic model), run
```
./runs/cvppp_train_ac.sh
```
Both scripts will train the respective models stagewise: the trained sequence length will gradually increase (5, 10, 15) while cutting the learning rate. ```schedules.lua``` contains the training schedule for both models.

### Using trained models for inference
Directory ```eval/``` contains the code to produce the final results for evaluation.
Put the pre-trained models of the checkpoint for evaluation into ```eval/cvppp/models/<MODEL-ID>```.
Then, run
```
th cvppp_main.lua -dataIn [DATA-IN] -dataOut [DATA-OUT] -modelIdx <MODEL-ID> 
```
where the parameters in backets should be replaced with your own values. The script will save the final predictions in ```[DATA-OUT]```.

## Citation
```
@inproceedings{Araslanov:2019:ACIS,
  title     = {Actor-Critic Instance Segmentation},
  author    = {Araslanov, Nikita and Rothkopf, Constantin and Roth, Stefan},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
