import os
from tabnanny import verbose 
from box.exceptions import BoxValueError, BoxKeyError
import yaml
from src.textsummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from typing import List



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file. 
    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"BoxValueError: {e}")
        raise e
    except BoxKeyError as e:
        logger.error(f"BoxKeyError: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error reading YAML file at {path_to_yaml}: {e}")
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: List, verbose: bool = True):
    """Creates directories if they do not exist.
    Args:
        path_to_directories (list[Path]): A list of directory paths to create.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")
