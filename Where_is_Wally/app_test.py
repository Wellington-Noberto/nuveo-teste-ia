import argparse
from src.utils import detect_object, generate_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cfg', '--cfg_file', help='Darknet model config file',
                    default='models/cfg/yolov4.cfg')
    ap.add_argument('-weights', '--weights_file', help='Darknet model weights file',
                    default='models/weights/yolov4_last.weights')
    ap.add_argument('-path', '--test_path', help='Path to the folder containing the test images',
                    default='data/01-WheresWally/TestSet')
    args = vars(ap.parse_args())

    results = detect_object(args['cfg_file'], args['weights_file'], args['test_path'])
    generate_csv(results)


if __name__ == '__main__':
    main()
