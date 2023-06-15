import os
import os.path as osp
from argparse import ArgumentParser

import cv2
from tqdm import tqdm

from card_detection import get_card_img
from card_recognizer import recognize
from dl_card_detection import DetectionModel


def process(args):
    input_files = os.listdir(args.input_dir)

    if args.use_dl_detection:
        detection_model = DetectionModel(args.dl_weights)

    for file in tqdm(input_files):
        src = osp.join(args.input_dir, file)
        img = cv2.imread(src)
        if args.use_dl_detection:
            output_img = detection_model.forward(img)
        else:
            output_img, _ = get_card_img(img)
        output_img = recognize(output_img)
        dst = osp.join(args.output_dir, file)
        cv2.imwrite(dst, output_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="../output/")
    parser.add_argument("--use-dl-detection", action='store_true')
    parser.add_argument("--dl-weights", default="./assets/best_model(1).pth", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process(args)
