# python src/analyze/objects-ground-dino/extract.py --gpu test-collection/selected-frames/02082013 jsonl --output test-collection/objects-ground-dino/{video_id}/{video_id}-objects-ground-dino.jsonl.gz
# maximum 256 classes/images

import sys
sys.path.append("")

import argparse
from typing import List, Dict, Any, Optional, Iterable
from itertools import islice

import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel

from src.analyze.extractor import BaseExtractor

import numpy as np
import pandas as pd
class FeatureExtractor(BaseExtractor):
    """Extracts objects from images using GroundingDINO model."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser): 
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
        parser.add_argument('--model_id', default="nielsr/siglip-base-patch16-224", help='Model ID model huggingface')
        parser.add_argument('--batch-size', type=int, default=4, help='Batch-size for processing')
        parser.add_argument('--max_objects', type=int, default=5, help='Take the first N objects has highest scores')
        super(FeatureExtractor, cls).add_arguments(parser)

    def __init__(self, args: argparse.Namespace):
        super(FeatureExtractor, self).__init__(args)
        self.device =  torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model_id = args.model_id

        self.model = SiglipModel.from_pretrained(self.model_id)
        # processor = AutoProcessor.from_pretrained(args.model_id)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        self.batch_size = args.batch_size
        self.max_objects = args.max_objects
        self.processor = None
        self.model = None
        print("args", args)

    def setup(self):
        if self.model is not None:
            return

        print(f'Loading SigLip model: {self.model_id}')
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = SiglipModel.from_pretrained(self.model_id)
        print(f'Loaded SigLip model.')

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f'{image_path}: Error loading image - {e}')
            return None



    def extract_iterable(self, image_paths: Iterable[str]):
        self.setup()


        def embed_siglip(batch_paths):

            batch_images = [self.load_image(path) for path in batch_paths]
            batch_images = [img for img in batch_images if img is not None]
            if not batch_images:
                return []
            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt")
                images_features = self.model.get_image_features(**inputs)
                # return image_features
                records = [{'feature_vector': f.tolist()} for f in images_features.cpu().numpy()]
                # print("records", records)
            return records

        batch = []
        for path in image_paths:
            batch.append(path)
            if len(batch) == self.batch_size:
                yield from embed_siglip(batch)
                batch = []
        
        if batch:  # Process any remaining images
            yield from embed_siglip(batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract immage with siglip.')
    FeatureExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = FeatureExtractor(args)
    extractor.run()