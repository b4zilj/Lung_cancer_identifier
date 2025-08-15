# repo.py
import os

def create_file(path):
    dir_name = os.path.dirname(path)
    if dir_name:  # only create directories if there's a folder in the path
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        f.write("")

def main():
    folders = [
        "data/raw/cancer",
        "data/raw/normal",
        "data/interim",
        "artifacts/models",
        "artifacts/reports",
        "src"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "params.yaml",
        "dvc.yaml",
        "Makefile",
        "src/template.py",
        "src/prepare.py",
        "src/dataset.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py",
        "src/predict.py"
    ]
    for file in files:
        create_file(file)

    print("✅ Repository structure created successfully.")

if __name__ == "__main__":
    main()