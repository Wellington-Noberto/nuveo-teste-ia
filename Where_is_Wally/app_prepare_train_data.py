import argparse
from src.utils import prepare_data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-train_folder', '--train_folder', help='Folder containing .json files',
                    default='data/01-WheresWally/TrainingSet')
    ap.add_argument('-dst', '--dst_folder', help='Folder containing .json files',
                    default='data/obj/train')
    args = vars(ap.parse_args())

    prepare_data(args['train_folder'], args['dst_folder'])


if __name__ == '__main__':
    main()


