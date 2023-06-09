import os
import os.path as osp
from argparse import ArgumentParser
from tqdm import tqdm
import cv2

from card_detection import get_card_img
from card_recognizer import recognize


def process(args):
    input_files = os.listdir(args.input_dir)
    for file in tqdm(input_files):
        
        src = osp.join(args.input_dir, file)
        img = cv2.imread(src)
        output_img = get_card_img(img)
        output_img = recognize(output_img)
        dst = osp.join(args.output_dir, file)
        cv2.imwrite(dst, output_img)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="../output/")
    args = parser.parse_args()
    process(args)
