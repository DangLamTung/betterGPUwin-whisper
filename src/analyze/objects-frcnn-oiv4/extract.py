# python src/analyze/objects-frcnn-oiv4/extract.py --gpu test-collection/selected-frames/02082013 jsonl --output test-collection/objects-frcnn-oiv4/{video_id}/{video_id}-objects-frcnn-oiv4.jsonl.gz

import sys
sys.path.append("")

import argparse
from functools import partial
import multiprocessing
import sys
from typing import List, Dict, Any, Optional

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F

from src.analyze.extractor import BaseExtractor
from src.analyze.utils import xyxy_to_yxyx


def load_image(image_path: str) -> Optional[torch.Tensor]:
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_tensor = F.to_tensor(img)
            # Normalize the tensor
            # img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.unsqueeze(0)
    except KeyboardInterrupt as e:
        raise e
    except:
        print(f'{image_path}: Error loading image')
        return None

    return img_tensor


def output_to_record(output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    # print("output", output)
    record = {
        # 'object_class_labels': output['labels'].tolist(),
        'object_class_names': [COCO_INSTANCE_CATEGORY_NAMES[i] for i in output['labels'].tolist()],
        'object_scores': output['scores'].tolist(),
        'object_boxes_yxyx': xyxy_to_yxyx(output['boxes']).tolist(), #Fix to yxyx 
        'detector': 'frcnn-oiv4',
    }
    # print("record", record)
    return record


def apply_detector(detector: torch.nn.Module, x: torch.Tensor) -> Optional[Dict[str, Any]]:
    if x is None:
        return None

    try:
        with torch.no_grad():
            y = detector(x)[0]
        record = output_to_record(y)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print(f'Error applying detector: {e}')
        return None

    return record


class ObjectDetectionExtractor(BaseExtractor):
    """Extracts objects from images using FasterRCNN ResNet50 FPN V2 detector."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
        super(ObjectDetectionExtractor, cls).add_arguments(parser)

    def __init__(self, args: argparse.Namespace):
        super(ObjectDetectionExtractor, self).__init__(args)
        self.detector = None
        self.device = torch.device(args.device)

    def setup(self):
        if self.detector is not None:
            return

        print(f'Loading detector: FasterRCNN ResNet50 FPN V2')
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)
        self.detector.to(self.device)
        self.detector.eval()
        print(f'Loaded detector.')

    def extract_one(self, image_path: str) -> Optional[Dict[str, Any]]:
        self.setup()  # lazy load model
        image = load_image(image_path)
        if image is not None:
            image = image.to(self.device)
        det = apply_detector(self.detector, image)
        return det

    def extract(self, image_paths: List[str]) -> List[Optional[Dict[str, Any]]]:
        records = map(self.extract_one, image_paths)
        records = list(records)
        return records

    def extract_iterable(self, image_paths: List[str], batch_size: int = 2):
        self.setup()

        for image_path in image_paths:
            image = load_image(image_path)
            if image is not None:
                image = image.to(self.device)
                out = self.detector(image)[0]
                record = output_to_record(out)
                yield record


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect Objects with PyTorch vision models.')
    ObjectDetectionExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = ObjectDetectionExtractor(args)
    extractor.run()