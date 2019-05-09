import argparse
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from logger.logger import setup_logging

JSON_FORMAT = ".json"
YML_FORMAT = ".yml"
DEFAULT_CONFIG_FILE_NAME = "config.json"
EXCLUTED_ARGS = ['device', 'resume', 'config']


class ConfigParser:
    '''
    Parser to load json or yaml based configuration files for pytorch
    '''
    def __init__(self, args: Dict, timestamp: bool=True):
        if "device" in args:
            os.environ["CUDA_VISIBLE_DEVICES"] = args['device']

        if 'resume' in args:
            self.resume: Union[Path, None] = Path(args['resume'])
            self.cfg_fname = self.resume.parent / DEFAULT_CONFIG_FILE_NAME
        elif 'config' in args:
            self.resume: Union[Path, None] = None
            self.cfg_fname = Path(args['config'])
        else:
            msg_no_cfg = "Configuration file need to be specified." \
                "add -c config.json for example."
            raise ValueError(msg_no_cfg)

        self._config = self.load(self.cfg_fname)
        self._update(args)

        # set save_dir where trained model and log will be saved
        save_dir = Path(self._config["trainer"]["save_dir"])
        expr_name = self._config["name"]
        if self.resume:
            time_stamp = self.resume.parent.stem if timestamp else ''
            self._save_dir = save_dir / "models" / expr_name / time_stamp
            self._log_dir = save_dir / "log" / expr_name / time_stamp

        else:
            time_stamp = datetime.now().strftime(r"%m%d_%H%M%S") if timestamp else ''
            self._save_dir = save_dir / "models" / expr_name / time_stamp
            self._log_dir = save_dir / "log" / expr_name / time_stamp

            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir.mkdir(parents=True, exist_ok=True)

            self.save(self._save_dir / DEFAULT_CONFIG_FILE_NAME)

        # configurations for logging module
        setup_logging(self.log_dir)
        self._log_level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def __str__(self):
        text = "PARAMETERS\n" + "-" * 20 + "\n"
        for (key, value) in self.__dict__.items():
            text += f"{key}:\t{value}" + "\n"
        return text

    def __getitem__(self, name: str):
        return self._config[name]

    def initialize(self, name: str, module: str, *args: List):
        """
        finds a function handle with the name given as 'type' in config.
        and returns instance initialized with corresponding given keyword args nad 'args'
        """
        module_cfg = self[name]
        return getattr(module, module_cfg["type"])(*args, **module_cfg["args"])

    @classmethod
    def parse(cls, args: Union[Dict, "argparse.Namespace"], timestamp: bool=True) -> "ConfigParser":
        """
        parse arguments from dict or argumentparser

        :param args: parameters to save.
        """
        if isinstance(args, argparse.Namespace):
            arguments = {}
            for (key, value) in vars(args).items():
                if value is not None:
                    arguments[key] = value
        else:
            arguments = args

        return cls(arguments, timestamp)

    def add(self, key: str, value: Any, update: bool = False) -> None:
        """
        add argument if key is not exists.
        it updates value when update=True
        """
        if self._config.get(key) is not None:
            if update and self.config.get(key) != value:
                print(f"Update parameter {key}: {self.get(key)} -> {value}")
                setattr(self._config, key, value)
        else:
            setattr(self._config, key, value)

    def _update(self, args: Dict) -> None:
        '''
        update config with args.
        loaded config file is OVERWRITTEN by args.
        '''
        assert isinstance(args, dict)
        for (key, value) in args.items():
            if key not in EXCLUTED_ARGS:
                self.add(key, value, update=True)

    def del_hparam(self, key: str) -> None:
        """
        delete argument if key is exists.
        """
        if hasattr(self._config, key):
            print(f"Delate parameter {key}")
            delattr(self._config, key)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        get value from key. this is wrapper for __getitem__
        """
        if hasattr(self._config, key):
            return getattr(self._config, key)

        return default

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> Dict:
        """
        parse config file of YML or JSON format.

        :param config_path: path to config file.
        :return dict of configurations.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)
        assert config_path.exists()

        with config_path.open() as handle:
            if config_path.suffix == JSON_FORMAT:
                config = json.load(handle, object_hook=OrderedDict)

            elif config_path.suffix == YML_FORMAT:
                yaml.add_constructor(  # For loading key and values with OrderedDict style.
                    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                    lambda loader, node: OrderedDict(loader.construct_pairs(node)),
                )
                config = yaml.load(handle, Loader=Loader)

            else:
                raise Exception("config_loader: config format is unknown.")

        return config

    def save(self, save_path: Union[str, Path]) -> None:
        """
        save config parameters to save_path.

        :param save_path: path to save config parameters.
        """
        if isinstance(save_path, str):
            save_path = Path(save_path)
        invalid_format_msg = "config_loader: config format is unknown."
        assert save_path.suffix in [JSON_FORMAT, YML_FORMAT], invalid_format_msg

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        with save_path.open("w") as handle:
            if save_path.suffix == JSON_FORMAT:
                json.dump(self._config, handle, indent=4, sort_keys=False)
            elif save_path.suffix == YML_FORMAT:
                handle.write(yaml.dump(self._config, default_flow_style=False))

    def get_logger(self, name: str, verbosity: int = 2):
        """
        get logger

        :param name:
        :param verbosity:
        :return: logger
        """
        msg_verbosity = f"verbosity option {verbosity} is invalid"
        assert verbosity in self._log_level, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self._log_level[verbosity])
        return logger

    @property
    def config(self):
        'get registerd configurations'
        return self._config

    @property
    def save_dir(self):
        'get the path to save directory.'
        return self._save_dir

    @property
    def log_dir(self):
        'get the path to logging directory.'
        return self._log_dir
