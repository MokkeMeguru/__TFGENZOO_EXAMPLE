#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig

import hydra


@hydra.main(config_path="./cond_conf/config.yaml")
def main(cfg: DictConfig):

    print(cfg.pretty())
    # task = Task(cfg)
    # task.train()
    # task.eval()


if __name__ == "__main__":
    main()
