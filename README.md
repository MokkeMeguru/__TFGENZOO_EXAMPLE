# What is this repository?
this repository is the introduction to use [TFGENZOO](https://github.com/MokkeMeguru/TFGENZOO)

# HOW TO RUN IT?

```
docker-compose build
docker-compose exec tfgenzoo /bin/bash
tmux
python glow.py
```

- you can see the result in outputs/< date-of-your-running >/< time-of-your-running >/glow.log
- you can see tensorboard log with some generated images and the learning curve.


example:
```text:glow.log
[2020-05-01 06:34:36,508][tensorflow][INFO] - z_f's shape             : [None, 3, 3, 16]
[2020-05-01 06:34:36,508][tensorflow][INFO] - log_det_jacobian's shape: (None,)
[2020-05-01 06:34:36,508][tensorflow][INFO] - z_aux's shape           : [None, 3, 3, 48]
[2020-05-01 06:34:36,774][tensorflow][INFO] - train data size: 48000
[2020-05-01 06:34:36,774][tensorflow][INFO] - valid data size: 12000
[2020-05-01 06:34:36,775][tensorflow][INFO] - test data size: 10000
[2020-05-01 06:34:36,914][tensorflow][INFO] - image size: (24, 24, 1)
[2020-05-01 06:34:36,930][tensorflow][INFO] - checkpoint will be saved at checkpoints/glow
[2020-05-01 06:37:09,151][tensorflow][INFO] - epoch 0: train_loss = 2.3704142570495605, valid_loss = 2.0007998943328857, saved_at = checkpoints/glow/ckpt-1
...
[2020-05-01 08:20:19,077][tensorflow][INFO] - epoch 63: train_loss = 1.3316023349761963, valid_loss = 1.4252331256866455, saved_at = checkpoints/glow/ckpt-64
[2020-05-01 08:21:01,998][tensorflow][INFO] - eval: nll 1.3283636569976807
```

![Learning Curve](/img/learning_curve.png)
![Generate Image](/img/image_generation.png)


# How to setup hyperparameter?
I use [Hydra](https://hydra.cc/) to manage hyper parameters.
So you can see the basic hyper parameter in conf/config.yaml


```yaml:config.yaml
defaults:
  - model: glow
  - dataset: mnist

check_model: true
batch_sizes: [128, 256, 256]
epochs: 64
```
