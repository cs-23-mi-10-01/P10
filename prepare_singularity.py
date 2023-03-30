import os
import shutil
from scripts import write


def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    move_dir = os.path.join(base_dir, "image_folder")
    all_files = os.listdir(base_dir)

    for i, file in enumerate(all_files):
        all_files[i] = os.path.join(base_dir, file)

    ignore_files = ("singularity.def", "environment.yml", "requirements.txt", "prepare_singularity.py")
    move_files = [file for file in all_files if not file.endswith(ignore_files)]
    move_files = [file for file in move_files if (os.path.sep + ".git" + os.path.sep) not in file]

    os.mkdir(move_dir)
    for file in move_files:
        shutil.move(file, move_dir)    


if __name__ == '__main__':
    main()


