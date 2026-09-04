from pathlib import Path

import kagglehub


DATASET = "mlg-ulb/creditcardfraud"
DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    path = kagglehub.dataset_download(DATASET, output_dir=DATA_DIR)
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()