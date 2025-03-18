# Benchmark

Please make sure you have downloaded the pre-trained weights following instructions in [install.md](https://github.com/tsvillaluenga/LEOD_FLW/blob/main/docs/install.md#pre-trained-weights).


## Evaluation

To evaluate a pre-trained model, set the `dataset`, `dataset.path`, and `checkpoint` fields accordingly.

```Bash
python val.py model=rnndet dataset=flw_dataset dataset.path=./datasets/flw_dataset/ \
  checkpoint="pretrained/WSOD/rvt-s-flwx0.01_ss-final.ckpt" \
  use_test_set=1 hardware.gpus=0 hardware.num_workers.eval=2 +experiment/flw_dataset="small.yaml" \
  batch_size.eval=4 model.postprocess.confidence_threshold=0.001 reverse=False tta.enable=False
```

We also support the `reverse` flag which tests the model on event sequences in the reverse-time order, and `tta.enable` which enables Test-Time Augmentation (model prediction ensembled over horizontal and temporal flip).


### Visualize the detection results

You can apply RVT to the entire event sequence and get continuous detection results as videos:

```Bash
python vis_pred.py model=rnndet dataset=flw_dataset dataset.path=./datasets/flw_dataset/ \
  checkpoint=xxx.ckpt +experiment/gen1="small.yaml" \
  model.postprocess.confidence_threshold=0.1 num_video=5 reverse=False
```

The mp4 files will be saved to `./vis/flw_dataset_rnndet_small/pred/`.
Below is one example running RVT-B on FLW_Dataset.
We pause the stream for 0.5s at every labeled frames to better see the detection results.
The predicted bboxes (with predicted cls and obj scores) are in green, while the GT bboxes are in black.
The frame index and timestamp are also shown at the top-left corner.

Sometimes could be a problem `num_video = 5`. If it is, increment the number of `num_video` drastically.


## Training

### Pre-train on data with limited annotations

Make sure to understand the data splits used in this project described in [install.md](https://github.com/tsvillaluenga/LEOD_FLW/blob/main/docs/install.md#data-splits).

We follow RVT for most of the settings, e.g., batch size, learning rate.
The biggest change we made is early stopping to prevent overfitting.

```Bash
python train.py model=rnndet hardware.gpus=0 dataset=flw_datasetx0.01_ss \
  +experiment/flw_dataset="small.yaml" training.max_steps=200000
```

Please refer to Appendix A.2 in the paper for different training steps we use under different ratio of data.
In general, we use 200k steps for 1% of the data, 300k steps for 2% of the data, and 400k steps for 5% and more data (because RVT trains for 400k steps on full data).

It is possible to get better results in this pre-training stage, as we did not tune the hyper-parameters very carefully for it.

### Generate pseudo labels for self-training

After obtaining a trained model (either trained on annotated data only, or after first-/second-round self-training), we can use it to generate pseudo labels on all unlabeled events for the next round of model training.
Note that, TTA ensemble, low-score filtering, and tracking-based post-processing are all done in this step.

The logic is simple, we run the model on all event sequences, and save the detected bboxes in the same format as the original dataset.
This way, we can re-use the same training code to train the model on the pseudo dataset.
Note that labels are saved as `.npy` files, while the events are soft-linked instead of copied to save storage.
The entire eval/test sets are also soft-linked.

```Bash
python predict.py model=pseudo_labeler dataset=flw_datasetx0.01_ss dataset.path=./datasets/flw_dataset/ \
  checkpoint="ned/WSOD/rvt-s-flw_datasetx0.01_ss-final.ckpt" \
  hardware.gpus=0 +experiment/flw_dataset="small.yaml" model.postprocess.confidence_threshold=0.01 \
  tta.enable=True save_dir=./datasets/pseudo_flw_dataset/flw_datasetx0.01_ss-1round/train
```

This script also run tests to ensure that the saved labels are in the correct format.
E.g., the saved labels should be the same on frames that are originally annotated.

### Evaluate the quality of pseudo labels

You may want to inspect the quality of the generated pseudo labels, e.g., their precision (AP) and recall (AR).
We provide scripts for doing this:

```Bash
python val_dst.py model=pseudo_labeler dataset=flw_datasetx0.01_ss \
  dataset.path=./datasets/pseudo_flw/flw_datasetx0.01_ss-1round checkpoint=1 \
  +experiment/flw_dataset="small.yaml" model.pseudo_label.obj_thresh=0.01 model.pseudo_label.cls_thresh=0.01
```

### Self-training on pseudo labels

Since our pseudo labels are saved in the same format as real labels, the training commands are almost the same.
You can simply copy an existing dataset config and change the path to the path of the pseudo dataset.
**Also remember to set `ratio` (WSOD) and `train_ratio` (SSOD) to -1**, so that we use all the pseudo labels.
We provide one such example at [flwx0.01_ss-1round.yaml](../config/dataset/flwx0.01_ss-1round.yaml).

```Bash
python train.py model=rnndet-soft hardware.gpus=0 dataset=flw_datasetx0.01_ss-1round \
  +experiment/flw_dataset="small.yaml" training.max_steps=150000 training.learning_rate=0.0005
```

Note that the model config here is `rnndet-soft`, which enables soft anchor assignment compared to vanilla `rnndet`.
Also, we only train for 150k steps and can use a larger learning rate.
This is because we have much denser annotations, which increases the effective batch size, and leads to faster convergence.

### Repeat the above steps

After the first round of self-training, you can repeat the process to generate pseudo labels for the next round of self-training.
From our experiments, the performance gain usually diminishes after the second round.
As discussed in Sec.4.4 of the paper, we found that the precision of pseudo labels is a good indicator of performance gain in the next round of self-training.

## Troubleshooting

In this project, there are two settings (WSOD, SSOD) each with several different data ratios (1%, 2%, 5%, 10%, full), and the base RVT model has different size variants (RVT-S, RVT-B).
It is very easy to make mistakes when running commands.
Here are some common error messages and their solutions:
- If you see lots of shape mismatch errors when trying to load a pre-trained weight, it might be:
  - You use the wrong model size (RVT-S vs RVT-B). Please check the `+experiment/flw_dataset="xxx.yaml"` field;
  - The dataset is wrong (e.g. testing a Gen1 trained model on 1Mpx). Please check the `dataset=xxx` field;
- If you see AssertionError saying the data/pseudo label formats are wrong, it is likely that you set the wrong data path or the wrong dataset config. Please check the `dataset=xxx` field.
