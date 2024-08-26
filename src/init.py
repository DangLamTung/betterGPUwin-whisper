import sys
sys.path.append("")

from pathlib import Path
import shutil
from src.Utils.utils import read_config

def _get_skel_dir():
    config = read_config("config/common_config.yaml")
    return config['skel_dir']

class InitCommand:
    def __init__(self, config_path:str = "config/common_config.yaml" ):
        self.config = read_config(config_path)

    def execute(self):

        directory = Path(self.config['collection_dir'])
        skel = self.config['skel_dir']
        if directory.exists():
            print(f"Directory {directory} already exists. Please choose a different path.")
            return 1
        
        shutil.copytree(skel, directory, dirs_exist_ok=True)
        print(f"Initialized collection in {directory.absolute()}")
        return 0

if __name__ == "__main__":
    command = InitCommand(config_path = "config/common_config.yaml")
    sys.exit(command.execute())
