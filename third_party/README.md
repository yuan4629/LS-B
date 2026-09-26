# Third-party code

This repository does not redistribute third-party code. `scripts/fetch_third_party.sh`
fetches it at pinned commits. The fetched files are not covered by the MIT license in
`LICENSE`.

| Code | Upstream | Commit | License | Local path |
|---|---|---|---|---|
| BiLoRA | <https://github.com/yifeiacc/BiLoRA> | `78ff950f44644bf248dc76531cda73d5f6b1ad57` | no license file upstream | `baselines/BiLoRA/` |
| brainnet (Brain Decodes Deep Nets, Yang, Gee and Shi, CVPR 2024) | <https://github.com/huzeyann/BrainDecodesDeepNets> | `8f16e48cbfb8acb041b3e984ba76bca08027b5e1` | CC BY-NC (upstream README) | `brainnet/` |

`brainnet_plmodel.patch` changes `brainnet/plmodel.py` so that it runs with current
torchmetrics and without the plotting dependencies when `draw=False`:

- R2Score is built in a way that works with old and new torchmetrics.
- pycortex and the plotting helpers are imported only when drawing.
- Channel clustering, which only feeds the visualisation, is skipped when `draw=False`.

The patch modifies CC BY-NC licensed code and is distributed under the same terms, not
under the MIT license in `LICENSE`.

`baselines/bilora_adapter/bilora_d2.py` contains an attention forward adapted from
`Attention_FFT.forward` in BiLoRA (`models/fft.py`); the comment above it says so.
