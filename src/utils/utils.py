import logging
import warnings
from math import floor
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import logging
import tqdm


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

        class TqdmLoggingHandler(logging.Handler):
            def __init__(self, level=logging.NOTSET):
                super().__init__(level)

            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.tqdm.write(msg)
                    self.flush()
                except Exception:
                    self.handleError(record)

        # logger.addHandler(TqdmLoggingHandler())

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "datamodule",
                "callbacks",
                "logger",
                "test_after_training",
                "seed",
                "name",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def batch_to_tensor(batch: list, included_y=True):
    if included_y:
        (sequences, categories) = zip(*batch)
        stack = [torch.stack(seq) for seq in sequences]
        new_sequences = torch.stack(stack, 1).transpose(1, 0)
        return new_sequences, torch.tensor(categories)
    sequences = zip(*batch)
    stack = [torch.stack(seq) for seq in sequences]
    new_sequences = torch.stack(stack, 1).transpose(1, 0)
    return new_sequences


def calc_embedding_size_recursive(conv_layers, input_size):
    if not len(conv_layers):
        return input_size
    conv_layer = conv_layers[0]
    res = floor(
        (input_size[0] + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
        conv_layer.stride[0] + 1)
    return calc_embedding_size_recursive(conv_layers[1:], (res, res))
