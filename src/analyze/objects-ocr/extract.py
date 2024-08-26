# python src/analyze/objects-frcnn-oiv4/extract.py --gpu test-collection/selected-frames/02082013 jsonl --output test-collection/objects-frcnn-oiv4/{video_id}/{video_id}-objects-frcnn-oiv4.jsonl.gz

import sys
sys.path.append("")

import argparse

import sys
from typing import List, Dict, Any, Optional, Iterable

from PIL import Image
import torch
from OcrReader import OcrReader
from src.analyze.extractor import BaseExtractor



class OCRFeatureExtractor(BaseExtractor):
    """Extracts OCR from images using Paddle OCR"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
        parser.add_argument('--language_detector', default="facebook/metaclip-b32-400m", help='Model ID for language detector')
        parser.add_argument('--language_threshold', type=float, default=0.2, help='Language detector threshold')
        parser.add_argument('--language_dict_file', default='src/analyze/objects-ocr/language_dict.json', help='Language path for ocr')

        super(OCRFeatureExtractor, cls).add_arguments(parser)

    def __init__(self, args: argparse.Namespace):
        super(OCRFeatureExtractor, self).__init__(args)
        self.ocr = None
        self.device = torch.device(args.device)
        self.language_detector=args.language_detector
        self.language_threshold=args.language_threshold
        self.language_dict_file=args.language_dict_file

    def setup(self):
        if self.ocr is not None:
            return

        self.ocr = OcrReader(json_path=self.language_dict_file,
                                  language_detector=self.language_detector,
                                  language_thresh=self.language_threshold,
                                  device=self.device,)
        print(f'Loaded ocr models.')

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f'{image_path}: Error loading image - {e}')
            return None

    def output_to_record(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        record = {
            'text': output['text'],
            'language': output['language'],
            'detector': 'paddel_ocr',
        }
        print("record", record)

        return record

    def extract_iterable(self, image_paths: List[str]):
        self.setup()

        for image_path in image_paths:
            image = self.load_image(image_path)
            if image is not None:
                out = self.ocr.get_text(image)
                record = self.output_to_record(out)
                yield record



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR recognition')
    OCRFeatureExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = OCRFeatureExtractor(args)
    extractor.run()