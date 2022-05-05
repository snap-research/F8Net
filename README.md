## F8Net<br><sub>Fixed-Point 8-bit Only Multiplication for Network Quantization (ICLR 2022 Oral)</sub>

[OpenReview](https://openreview.net/forum?id=_CfpJazzXT2) | [arXiv](https://arxiv.org/abs/2202.05239) | [PDF](https://arxiv.org/pdf/2202.05239.pdf) | [Model Zoo](#f8net-model-zoo) | [BibTex](#citation)


<img src="https://user-images.githubusercontent.com/25779973/164990377-bb692b26-4c7c-41bb-be8c-d91bdc6e8715.png" width=100%/>


PyTorch implementation of neural network quantization with fixed-point 8-bit only multiplication. <br>
>[F8Net: Fixed-Point 8-bit Only Multiplication for Network Quantization](https://openreview.net/forum?id=_CfpJazzXT2)<br>
>[Qing Jin](https://scholar.google.com/citations?user=X9iggBcAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Jian Ren](https://alanspike.github.io/)<sup>1</sup>, [Richard Zhuang](https://www.linkedin.com/in/richard-zhuang-82ba504/)<sup>1</sup>, [Sumant Hanumante](https://www.linkedin.com/in/sumant-hanumante-3a698123/)<sup>1</sup>, [Zhengang Li](https://scholar.google.com/citations?user=hH1Oun0AAAAJ&hl=en)<sup>2</sup>, [Zhiyu Chen](https://vlsi.rice.edu/authors/zhiyu/)<sup>3</sup>, [Yanzhi Wang](https://coe.northeastern.edu/people/wang-yanzhi/)<sup>2</sup>, [Kaiyuan Yang](https://vlsi.rice.edu/authors/admin/)<sup>3</sup>, [Sergey Tulyakov](http://www.stulyakov.com/)<sup>1</sup>  
><sup>1</sup>Snap Inc., <sup>2</sup>Northeastern University, <sup>3</sup>Rice University<br>
>ICLR 2022 Oral.


<details>
  <summary>
  <font size="+1">Overview</font>
  </summary>
    Neural network quantization implements efficient inference via reducing the weight and input precisions. Previous methods for quantization can be categorized as simulated quantization, integer-only quantization, and fixed-point quantization, with the former two involving high-precision multiplications with 32-bit floating-point or integer scaling. In contrast, fixed-point models can avoid such high-demanding requirements but demonstrates inferior performance to the other two methods. In this work, we study the problem of how to train such models. Specifically, we conduct statistical analysis on values for quantization and propose to determine the fixed-point format from data during training with some semi-empirical formula. Our method demonstrates that high-precision multiplication is not necessary for the quantized model to achieve comparable performance as their full-precision counterparts.
</details>


## Getting Started

<details>
  <summary>Requirements</summary>

   1. Please check the [requirements](/requirements.txt) and download packages.

   2. Prepare ImageNet-1k data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet), and create a softlink to the ImageNet data path to data under current the code directory (`ln -s /path/to/imagenet data`).

</details>

<details><summary>Model Training</summary><blockquote>

  <details><summary> Conventional training </summary><blockquote>
  
  * We train the model with the file [distributed_run.sh](/distributed_run.sh) and the command 
    ```
    bash distributed_run.sh /path/to/yml_file batch_size
    ```
  * We set `batch_size=2048` for conventional training of floating-/fixed-point ResNet18 and MobileNet V1/V2.
  * Before training, please update the `dataset_dir` and `log_dir` arguments in the yaml files for training the floating-/fixed-point models.
  * To train the floating-point model, please use the yaml file `***_floating_train.yml` in the `conventional` subfolder under the corresponding folder of the model.
  * To train the fixed-point model, please first train the floating-point model as the initialization. Please use the yaml file `***_fix_quant_train.yml` in the `conventional` subfolder under the corresponding folder of the model. Please make sure the argument `fp_pretrained_file` directs to the correct path for the corresponding floating-point checkpoint. We also provide our pretrained floating-point models in the [Model Zoo](#f8net-model-zoo) below.
</blockquote></details>

<details><summary> Tiny finetuning </summary><blockquote>

* We finetune the model with the file [run.sh](/run.sh) and the command 
    ```
    bash run.sh /path/to/yml_file batch_size
    ```
* We set `batch_size=128` and use one GPU for tiny-finetuning of fixed-point ResNet18/50.

* Before fine-tuning, please update the `dataset_dir` and `log_dir` arguments in the yaml files for finetuning the fixed-point models.

* To finetune the fixed-point model, please use the yaml file `***_fix_quant_***_pretrained_train.yml` in the `tiny_finetuning` subfolder under the corresponding folder of the model. For model pretrained with [`PytorchCV`](https://pypi.org/project/pytorchcv/) (Baseline of ResNet18 and Baseline#1 of ResNet50), the floating-point checkpoint will be downloaded automatically during code running. For the model pretrained by [`Nvidia`](https://catalog.ngc.nvidia.com/orgs/nvidia/models/resnet50_pyt_amp/files) (Baseline#2 of ResNet50), please download the checkpoint first and make sure the argument `nvidia_pretrained_file` directs to the correct path of this checkpoint.

</blockquote></details>

</blockquote></details>

<details>
<summary>Model Testing</summary>
  
* We test the model with the file [run.sh](/run.sh) and the command 
    ```
    bash run.sh /path/to/yml_file batch_size
    ```
* We set `batch_size=128` and use one GPU for model testing.

* Before testing, please update the `dataset_dir` and `log_dir` arguments in the yaml files. Please update the argument `integize_file_path` and `int_op_only_file_path` arguments in the yaml files `***_fix_quant_test***_integize.yml` and `***_fix_quant_test***_int_op_only.yml`, respectively. Please also update other arguments like `nvidia_pretrained_file` if necessary (even if they are not used during testing).

* We use the yaml file `***_floating_test.yml` for testing the floating-point model; `***_fix_quant***_test.yml` for testing the fixed-point model with the same setting as during training/tiny-finetuning; `***_fix_quant***_test_int_model.yml` for testing the fixed-point model on GPU with all quantized weights, bias and inputs implemented with integers (but with `float` dtype as GPU does not support integer operations) and use the original modules during training (e.g. with batch normalization layers); `***_fix_quant***_test_integize.yml` for testing the fixed-point model on GPU with all quantized weights, bias and inputs implemented with integers (but with `float` dtype as GPU does not support integer operations) and a new equivalent model with only convolution, pooling and fully-connected layers; `***_fix_quant***_test_int_op_only.yml` for testing the fixed-point model on CPU with all quantized weights, bias and inputs implemented with integers (with `int` dtype) and a new equivalent model with only convolution, pooling and fully-connected layers. Note that the accuracy from the four testing files can differ a little due to numerical error.

</details>


<details>
  <summary>Model Export</summary>
  
* We export fixed-point model with integer weights, bias and inputs to run on GPU and CPU during model testing with `***_fix_quant_test_integize.yml` and `***_fix_quant_test_int_op_only.yml` files, respectively.

* The exported onnx files are saved to the path given by the arguments `integize_file_path` and `int_op_only_file_path`.

</details>




## F8Net Model Zoo

All checkpoints and onnx files are available at **[here](https://drive.google.com/drive/folders/1lYWPj9TB-c50lIxXlYWbCfpF5pSAP0fc?usp=sharing)**.

**Conventional**

| Model | Type | Top-1 Acc.<sup>a</sup> | Checkpoint |
| :--- | :---: | :---: | :---: |
| ResNet18 | FP<br>8-bit | 70.3<br>71.0 | [`Res18_32`](https://drive.google.com/file/d/1BxRPKr7SRQmrRdJt1oUrRxzjas65ItQN/view?usp=sharing)<br>[`Res18_8`](https://drive.google.com/file/d/1U93c7QLHs0Ww_93yY1msbRsghZTGaruG/view?usp=sharing) |
| MobileNet-V1 | FP<br>8-bit | 72.4<br>72.8 | [`MBV1_32`](https://drive.google.com/file/d/14zeH0HLUS8UN7RKDyKWMHPKzXKa6mesp/view?usp=sharing)<br>[`MBV1_8`](https://drive.google.com/file/d/1Q89sIqR2HrCcEOOcLrKl8emcippkT6p3/view?usp=sharing) |
| MobileNet-V2b | FP<br>8-bit | 72.7<br>72.6 | [`MBV2b_32`](https://drive.google.com/file/d/1OYz0CkLLQ2JV-X666HxiBVFAbJ3ojWIw/view?usp=sharing)<br>[`MBV2b_8`](https://drive.google.com/file/d/1YbDKgxHBFrBLhsZ4GJoL5R4sm5L8BT0p/view?usp=sharing) |

**Tiny Finetuning**

| Model | Type | Top-1 Acc.<sup>a</sup> | Checkpoint |
| :--- | :---: | :---: | :---: |
| ResNet18 | FP<br>8-bit | 73.1<br>72.3 | `Res18_32p`<br>[`Res18_8p`](https://drive.google.com/file/d/1L2vziGb5_OCjlA-cAoUk-54jA9BA-spN/view?usp=sharing) |
| ResNet50b (BL#1) | FP <br>8-bit | 77.6<br>77.6 | `Res50b_32p`<br>[`Res50b_8p`](https://drive.google.com/file/d/1YHe7MB4JpG75uo8GMpCxwsVHvAJflXF0/view?usp=sharing) |
| ResNet50b (BL#2) | FP <br>8-bit | 78.5<br>78.1 | [`Res50b_32n`](https://catalog.ngc.nvidia.com/orgs/nvidia/models/resnet50_pyt_amp/files)<br>[`Res50b_8n`](https://drive.google.com/file/d/1WU_ccesykRVKp9ntEDn_mieYTW-wAkkN/view?usp=sharing) |

<sup>a</sup> The accuracies are obtained from the inference step during training. Test accuracy for the final exported model might have some small accuracy difference due to numerical error.


## Technical Details

The main techniques for neural network quantization with 8-bit fixed-point multiplication includes the following:
  * Quantized methods/modules including determining fixed-point formats from statistics or by grid-search, fusing convolution and batch normalization layers, and reformulating PACT with fixed-point quantization are implemented in [`models/fix_quant_ops`](/models/fix_quant_ops.py).
  * Clipping-level sharing and private fractional length for residual blocks are implemented in the ResNet ([`models/fix_resnet`](/models/fix_resnet.py)) and MobileNet V2 ([`models/fix_mobilenet_v2`](/models/fix_mobilenet_v2.py)).


## Acknowledgement
This repo is based on [AdaBits](https://github.com/deJQK/AdaBits).


## Citation
If our code or models help your work, please cite our paper:
```bibtex
@inproceedings{
  jin2022fnet,
  title={F8Net: Fixed-Point 8-bit Only Multiplication for Network Quantization},
  author={Qing Jin and Jian Ren and Richard Zhuang and Sumant Hanumante and Zhengang Li and Zhiyu Chen and Yanzhi Wang and Kaiyuan Yang and Sergey Tulyakov},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=_CfpJazzXT2}
}
```
