import argparse
import zipfile
from pathlib import Path

from .config import RAW_DIR


DEFAULT_KAGGLE_DATASET = 'nicolejyt/facialexpressionrecognition'
DEFAULT_OUTPUT = RAW_DIR / 'fer2013.csv'


def download_fer2013(dataset=DEFAULT_KAGGLE_DATASET, output_path=DEFAULT_OUTPUT, force=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(f'{output_path} already exists. Use --force to replace it.')

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            'Kaggle CLI is not installed. Install it with: pip install kaggle'
        ) from exc

    download_dir = output_path.parent
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=download_dir, force=True, quiet=False)

    zip_files = sorted(download_dir.glob('*.zip'), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f'No downloaded zip file found in {download_dir}')

    with zipfile.ZipFile(zip_files[0]) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if not csv_members:
            raise FileNotFoundError('Downloaded archive does not contain a CSV file.')
        member = next((name for name in csv_members if Path(name).name == output_path.name), csv_members[0])
        with archive.open(member) as source, output_path.open('wb') as target:
            target.write(source.read())

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download FER2013 dataset from Kaggle.')
    parser.add_argument('--dataset', default=DEFAULT_KAGGLE_DATASET, help='Kaggle dataset slug.')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='Output CSV path.')
    parser.add_argument('--force', action='store_true', help='Replace existing output file.')
    args = parser.parse_args()

    output_path = download_fer2013(args.dataset, args.output, force=args.force)
    print(f'Dataset saved to {output_path}')


if __name__ == '__main__':
    main()
