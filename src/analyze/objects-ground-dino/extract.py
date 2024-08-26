# python src/analyze/objects-ground-dino/extract.py --gpu test-collection/selected-frames/02082013 jsonl --output test-collection/objects-ground-dino/{video_id}/{video_id}-objects-ground-dino.jsonl.gz
# maximum 256 classes/images

import sys
sys.path.append("")

import argparse
from typing import List, Dict, Any, Optional, Iterable
from itertools import islice

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from src.analyze.extractor import BaseExtractor
from src.analyze.utils import xyxy_to_yxyx
# from src.Utils.utils import timeit

class ObjectDetectionExtractor(BaseExtractor):
    """Extracts objects from images using GroundingDINO model."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
        parser.add_argument('--model_id', default='IDEA-Research/grounding-dino-base', help='Model ID for GroundingDINO')
        parser.add_argument('--text_prompt_file', default='src/analyze/objects-ground-dino/label.txt', help='Text prompt (labels) path for object detection')
        parser.add_argument('--box_threshold', type=float, default=0.35, help='Box threshold for post-processing')
        parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold for post-processing')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch-size for processing')
        parser.add_argument('--max_objects', type=int, default=5, help='Take the first N objects has highest scores')
        super(ObjectDetectionExtractor, cls).add_arguments(parser)

    def __init__(self, args: argparse.Namespace):
        super(ObjectDetectionExtractor, self).__init__(args)
        self.device = torch.device(args.device)
        self.model_id = args.model_id
        self.text_prompt_file = args.text_prompt_file
        with open(self.text_prompt_file, 'r') as file:
            self.text_prompt = file.read()
        print("self.text_prompt", self.text_prompt)

        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.batch_size = args.batch_size
        self.max_objects = args.max_objects
        self.processor = None
        self.model = None
        print("args", args)

    def setup(self):
        if self.model is not None:
            return

        print(f'Loading GroundingDINO model: {self.model_id}')
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        print(f'Loaded GroundingDINO model.')

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f'{image_path}: Error loading image - {e}')
            return None

    def output_to_record(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        n = min(self.max_objects, len(output['labels']))
        record = {
            'object_class_names': output['labels'][:n],
            'object_scores': output['scores'].cpu().tolist()[:n],
            'object_boxes_yxyx': xyxy_to_yxyx(output['boxes']).cpu().tolist()[:n],
            'detector': 'grounding-dino',
        }
        print("record", record)

        return record

    def extract_iterable(self, image_paths: Iterable[str]):
        self.setup()

        def process_batch(batch_paths):
            batch_images = [self.load_image(path) for path in batch_paths]
            batch_images = [img for img in batch_images if img is not None]
            if not batch_images:
                return []

            inputs = self.processor(images=batch_images, text=[self.text_prompt] * len(batch_images), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[img.size[::-1] for img in batch_images]
            )

            return [self.output_to_record(result) for result in results]

        batch = []
        for path in image_paths:
            batch.append(path)
            if len(batch) == self.batch_size:
                yield from process_batch(batch)
                batch = []
        
        if batch:  # Process any remaining images
            yield from process_batch(batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect Objects with GroundingDINO model.')
    ObjectDetectionExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = ObjectDetectionExtractor(args)
    extractor.run()