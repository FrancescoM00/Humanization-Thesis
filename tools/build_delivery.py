"""Build the professor's ZIP from versioned files, without models or environments."""
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'dist/Humanization.zip'
    target.parent.mkdir(exist_ok=True)
    if (ROOT/'.git').exists():
        files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    else:
        files=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()]
    forbidden={'.git','.venv','.cache','__pycache__'}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in files:
            if not name:continue
            file=ROOT/name
            if not file.is_file() or forbidden.intersection(Path(name).parts) or name=='.env':continue
            if name.startswith(('results/new_runs/','data/models/','dist/')):continue
            archive.write(file,'Humanization/'+name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise RuntimeError('The generated ZIP is corrupt')
        print(f'{target}: {len(archive.namelist())} files, {target.stat().st_size/1024**2:.1f} MiB')


if __name__=='__main__':main()
