import argparse
from src.utils import convert_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-folder', '--folder_path', help='Folder containing .json files',
                    default='data/01-WheresWally/TrainingSet')
    args = vars(ap.parse_args())

    convert_json(args['folder_path'])


if __name__ == '__main__':
    main()
