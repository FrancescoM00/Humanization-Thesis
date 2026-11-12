"""Utilità condivise per leggere YAML e percorsi negli script degli esperimenti."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import os
import yaml
from .paths import project_root

os.environ.setdefault('MPLCONFIGDIR',str(project_root()/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(project_root()/'.cache'))


def load(path):
    path=Path(path)
    if not path.is_absolute(): path=project_root()/path
    value=yaml.safe_load(path.read_text())
    if not isinstance(value,dict): raise ValueError('The YAML configuration must be a mapping')
    return value


def path(value):
    value=Path(value)
    return value if value.is_absolute() else project_root()/value


def output(value):
    value=path(value).resolve()
    if value.is_relative_to((project_root()/'results/published').resolve()):
        raise ValueError('New outputs must not overwrite results/published')
    value.mkdir(parents=True,exist_ok=True)
    return value


def record(directory,config,mode,**details):
    directory=output(directory)
    result={'mode':mode,'models_executed':mode=='compute',**details}
    (directory/'execution.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (directory/'configuration.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    return result


def copy_reference(source,destination):
    source=path(source);destination=output(destination)
    if source.resolve()==destination.resolve(): raise ValueError('Replay needs a separate destination')
    shutil.copytree(source,destination,dirs_exist_ok=True)
    return destination


def model_config(config,directory):
    """Prepara gli adattatori dei modelli usando esattamente le impostazioni configurate."""
    filename=output(directory)/'model_presets.json'
    filename.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    return filename
