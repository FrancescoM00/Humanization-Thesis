"""Percorsi relativi al progetto di ricerca, autonomo e utilizzabile dopo l’estrazione."""
from pathlib import Path


def project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent/'pyproject.toml').is_file() and (parent/'configs').is_dir():
            return parent
    raise RuntimeError('Open the extracted project containing pyproject.toml and configs/')


def resolve_path(root,value):
    """Risolve una sola volta il percorso dichiarato, senza tradurre i vecchi nomi delle cartelle."""
    value=Path(value)
    result=(value if value.is_absolute() else Path(root)/value).resolve()
    if any(a==b=='published' for a,b in zip(result.parts,result.parts[1:])):
        raise ValueError('The configured path contains duplicated published directories')
    return result
