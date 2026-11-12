"""Prepara una consegna autonoma con inclusioni esplicite, senza copiare l’intera cartella di lavoro Git."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES=('README.md','requirements.txt','pyproject.toml','THIRD_PARTY_NOTICES.txt',
            'Humanization.code-workspace','.gitignore','.gitattributes','.env.example')
INCLUDE_TREES=('.vscode','configs','requirements','src','experiments','tests','tools','docs',
               'data/raw/mgtbench','data/raw/opai','data/processed','data/resources','results/published')
EXCLUDED_PARTS={'.git','.venv','.cache','.pytest_cache','__pycache__','generated'}
EXCLUDED_SUFFIXES={'.pyc','.pyo','.log','.safetensors','.onnx','.pt','.bin'}


def delivery_files(root=ROOT):
    root=Path(root);files={root/name for name in ROOT_FILES if (root/name).is_file()}
    for name in INCLUDE_TREES:
        tree=root/name
        if tree.exists():files.update(p for p in tree.rglob('*') if p.is_file())
    return sorted(p for p in files if not EXCLUDED_PARTS.intersection(p.relative_to(root).parts)
                  and p.suffix not in EXCLUDED_SUFFIXES and not p.name.endswith('_progress.json')
                  and p.name!='.env'
                  and (p.suffix.lower()!='.md' or p==root/'README.md'))


def validate_portability(root,files):
    retired=re.compile((b'francescom00/'+b'humanization')+rb'(?!-thesis)',re.I)
    workspace=(b'/workspace/'+b'Humanization')
    violations=[]
    for p in files:
        content=p.read_bytes()
        if retired.search(content) or workspace in content:
            violations.append(p.relative_to(root).as_posix())
    if violations:raise ValueError('Non-portable delivery references in: '+', '.join(violations))


def build(root=ROOT,target=None):
    root=Path(root);target=Path(target) if target else root/'dist/Humanization.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    files=delivery_files(root);validate_portability(root,files)
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:archive.write(p,'Humanization/'+p.relative_to(root).as_posix())
        archive.writestr('Humanization/DELIVERY_MANIFEST.json',json.dumps({'path_base':'project_root','files_sha256':hashes},indent=2)+'\n')
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise RuntimeError('The generated ZIP is corrupt')
        assert not any('/archive/' in name for name in archive.namelist())
        markdown=[name for name in archive.namelist() if name.lower().endswith('.md')]
        if markdown!=['Humanization/README.md']:
            raise RuntimeError('La consegna deve contenere soltanto README.md come file Markdown')
    print(f'{target}: {len(files)} project files, {target.stat().st_size/1024**2:.1f} MiB')
    return target


if __name__=='__main__':build()
