"""Project experiment; configuration is read from configs/."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from humanization.cli import main
if __name__=='__main__':main('01')
