# Datasets and pretrained weights

This repository does not redistribute any dataset, image, fMRI recording or model
weight. Obtain each item from its source under that source's terms and place it at the
path shown (paths are relative to the repository root; all are ignored by git).

## Continual-learning benchmarks

| Dataset | Source | Terms | Expected path | Check |
|---|---|---|---|---|
| ImageNet-R | <https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar> (2,191,079,936 bytes) | see the ImageNet-R release | `data/imagenet-r/<class>/<image>` | train/test split below |
| CUB-200-2011 (APER split) | `cub.zip` from the RevisitingCIL/APER instructions (<https://github.com/LAMDA-CL/RevisitingCIL>), Google Drive id `1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb`; original data: CaltechDATA record `65de6-vp158` | non-commercial research and education only | `data/cub/train/<class>/`, `data/cub/test/<class>/` | `cub.zip` sha256 `876c3c857e74ab0b06f39af11f69e415330a8b5e592c35bd732e9ba09c1d1df5`; `baselines/bilora_adapter/cub_data.py` checks image counts (9430 / 2358) and the sha256 of the sorted file lists |
| CIFAR-100 | downloaded by torchvision on first use | cite Krizhevsky (2009) | `data/cifar-100-python/` | torchvision md5 `eb9058c3a382ffc7106e4002c42a8d85` |

### ImageNet-R split

The split is generated, not downloaded:

```bash
python make_inr_split.py --inr_root data/imagenet-r --out data/imagenet-r_split
```

It uses only file names (seed 1997, 20% test per class) and should produce 24,002 training
and 5,998 test images with

| File | sha256 |
|---|---|
| `data/imagenet-r_split/train.txt` | `490c9955446503f9d4923f9af77776c7cb644f0254e752c1546419b0a28a7aa4` |
| `data/imagenet-r_split/test.txt` | `2959a2388b1809b0c2ed546ef4ef1d94608d03bfcd9eeac683399980690c95f0` |

`python make_inr_split.py --selftest` runs a self-test on a temporary fake directory.

## fMRI data for the brain encoder

| Dataset | Source | Terms | Expected path |
|---|---|---|---|
| Algonauts 2023 challenge data, subject 1 (a subset of the Natural Scenes Dataset, NSD) | request through the Algonauts 2023 challenge page, <http://algonauts.csail.mit.edu/challenge.html> | non-commercial research and education; no redistribution. NSD terms: <https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions> (cite Allen et al., 2021, and acknowledge NSF IIS-1822683 and IIS-1822929) | `subj01/` with `training_split/`, `test_split/`, `roi_masks/` |

Expected contents: 9,841 training images, 159 test images, 19,004 left-hemisphere and
20,544 right-hemisphere vertices.

| File | sha256 |
|---|---|
| `subj01/training_split/training_fmri/lh_training_fmri.npy` | `70a76c9d0cf8d89a0d29970bddf3a44620ce5089cf8b1f06217b93eba26e135c` |
| `subj01/training_split/training_fmri/rh_training_fmri.npy` | `c2c3ee8871da553682b8810f315425ad3aa31b750a61c1a354f3a5a0ddb83405` |

The brain encoder (`brainnet`, fetched by `scripts/fetch_third_party.sh`) also downloads the
fsaverage7 surface through nilearn (0.13.1) on first use.

## Pretrained backbones

All three are ViT-B/16 checkpoints released under Apache-2.0. Download them to
`pretrained/` with the file names below.

| File | Source | sha256 |
|---|---|---|
| `pretrained/vit_b16_augreg_in21k.npz` (AugReg, ImageNet-21k) | <https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz> | `a157033e309d33e79cbff2408dd77b4458014fcedc46f0ad853e2d08ecb493f6` |
| `pretrained/ibot_vitb16.pth` (iBOT, teacher checkpoint) | <https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth> | `734b9c49f46cd330d740f6c35d6a83517aab770a3efd86d7f1175ced582c9b7f` |
| `pretrained/dino_vitbase16_pretrain.pth` (DINO) | <https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth> | `bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e` |

Verify with `sha256sum pretrained/*`. If a weights file passed on the command line does not
exist, the loader raises instead of downloading other weights.

## Third-party code

| Code | Source (pinned) | License |
|---|---|---|
| BiLoRA | <https://github.com/yifeiacc/BiLoRA> @ `78ff950f44644bf248dc76531cda73d5f6b1ad57` | no license file upstream; not redistributed |
| brainnet (Brain Decodes Deep Nets) | <https://github.com/huzeyann/BrainDecodesDeepNets> @ `8f16e48cbfb8acb041b3e984ba76bca08027b5e1` | CC BY-NC (upstream README); not redistributed |

`scripts/fetch_third_party.sh` clones both at these commits and applies
`third_party/brainnet_plmodel.patch`.
