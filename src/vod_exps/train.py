import multiprocessing
import os
import pathlib
import sys

import hydra
import lightning as L
import omegaconf
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from vod_workflows.utils.distributed import is_gloabl_zero

try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    logger.debug("Could not set multiprocessing start method to `forkserver`")
import torch
import vod_models
from vod_exps import recipes
from vod_exps import utils as exp_utils
from vod_tools import pretty

from .hydra import hyra_conf_path, register_omgeaconf_resolvers
from .structconf import Experiment

register_omgeaconf_resolvers()


@hydra.main(config_path=hyra_conf_path(), config_name="main", version_base="1.3")
def cli(hydra_config: DictConfig) -> None:
    """Training CLI.."""
    run_exp(hydra_config)


def run_exp(hydra_config: DictConfig) -> torch.nn.Module:
    """Train a ranker for a retrieval task."""
    if hydra_config.load_from is not None:
        logger.info(f"Loading checkpoint from `{hydra_config.load_from}`")
        checkpoint_dir = hydra_config.load_from
        cfg_path = pathlib.Path(hydra_config.load_from, "config.yaml")
        hydra_config = omegaconf.OmegaConf.load(cfg_path)  # type: ignore
    else:
        checkpoint_dir = None

    if is_gloabl_zero():
        pretty.pprint_config(hydra_config, exclude=["search_defaults"])

    # Set training context (env variables, muliprocessing, omp_threads, etc.)
    logger.debug(f"Setting environment variables from config: {hydra_config.env}")
    os.environ.update({k: str(v) for k, v in hydra_config.env.items()})
    exp_utils.set_training_context()
    logger.info(f"Experiment directory: {pathlib.Path().absolute()}")

    # Init the Fabric, log the hyperparameters
    logger.debug(f"Instantiating Lighning's Fabric <{hydra_config.fabric._target_}>")
    fabric: L.Fabric = instantiate(hydra_config.fabric)
    fabric.launch()

    # Init the model, optimizer and scheduler
    logger.debug(f"Instantiating model <{hydra_config.model._target_}> (seed={hydra_config.seed})")
    if hydra_config.seed is not None:
        fabric.seed_everything(hydra_config.seed)
    module: vod_models.Ranker = instantiate(hydra_config.model)
    optimizer = module.get_optimizer()
    scheduler = module.get_scheduler(optimizer)

    # Log config & setup logger
    _customize_logger(fabric=fabric)
    if fabric.is_global_zero:
        exp_utils.log_config(
            config=hydra_config,
            exp_dir=pathlib.Path(),
            extras={"model_stats": exp_utils.get_model_stats(module)},
            fabric=fabric,
        )

    # Setup the optimizer and fabric
    logger.debug("Setting up model and optimizer using `lightning.Fabric`")
    fabric_module, fabric_optimizer = fabric.setup(module, optimizer)

    # Train the model
    return recipes.periodic_training(
        module=fabric_module,
        optimizer=fabric_optimizer,
        scheduler=scheduler,
        fabric=fabric,
        config=Experiment.parse(hydra_config),
        resume_from_checkpoint=checkpoint_dir,
    )


if __name__ == "__main__":
    try:
        cli()
        logger.info(f"Success. Experiment logged to {pathlib.Path().absolute()}")
    except Exception as exc:
        logger.warning(f"Failure. Experiment logged to {pathlib.Path().absolute()}")
        raise exc


def _customize_logger(fabric: L.Fabric) -> None:
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<yellow>rank</yellow>:<yellow>{extra[rank]}/{extra[world_size]}</yellow> - <level>{message}</level>"
    )
    if fabric.world_size == 1:
        return

    logger.configure(extra={"rank": 1 + fabric.global_rank, "world_size": fabric.world_size})  # Default values
    logger.remove()
    logger.add(sys.stderr, format=logger_format)
