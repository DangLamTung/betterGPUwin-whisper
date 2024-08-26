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
import json
# from src.analyze.extractor import BaseExtractor
from transformers import pipeline
import glob
class FeatureWhisper:
    """Extracts objects from images using GroundingDINO model."""

    @classmethod
    def add_arguments(cls, parser):
        """ Add arguments to the parser. """

        parser.add_argument('--gpu', default=False, action='store_true', help='use the GPU if available')
        parser.add_argument('--batch-size', type=int, default=1, help='flush every N records extracted')

        parser.add_argument('input_images', type=str, help='images to be processed.'
            'Can be a directory or a file with a list of images.'
            'Each line of the list must be in the format: [[<video_id>\\t]<image_id>\\t]<image_path>\n'
            'If <video_id> is specified, contiguos images with the same <video_id> will be grouped together in the output files.'
            'If <image_id> is not specified, an incremental number will be used instead.')

        subparsers = parser.add_subparsers(dest='output_type')

        file_parser = subparsers.add_parser('jsonl', help='save results to gzipped JSONL files')
        file_parser.add_argument('-o', '--output', type=str, help='output path template, where "{video_id}" will be replaced by the video id.')

        hdf5_parser = subparsers.add_parser('hdf5', help='save results to HDF5 files')
        hdf5_parser.add_argument('-n', '--features-name', default='generic', help='identifier of feature type')
        hdf5_parser.add_argument('-o', '--output', type=str, help='output path template, where "{video_id}" will be replaced by the video id.')



   

    def __init__(self, args: argparse.Namespace):
        # super(BaseExtractor, self).__init__(args)
        self.device =  torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # self.model_id = args.model_id
        self.input = args.input_images
        self.output = args.output
        # self.features-name = args.features-name
        self.model = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-tiny",device = self.device )
   
       
        # print("args", args)

    def setup(self):
        if self.model is not None:
            return


    def load_audio(self, image_path: str):
        # try:
        return image_path
        # except Exception as e:
        #     print(f'{image_path}: Error loading image - {e}')
        #     return None


    def extract(self):
        

        files = sorted(glob.glob(self.input+"/*.wav"))
        # print(files)
        record = []
        with torch.no_grad():
            for f in files:
                output = self.model(f)['text']
                record.append({'text': output,
                                'language': "VN",
                                'detector': 'pho_whisper'})
        # print(record)
        

        with open(self.output, 'w') as outfile:
            for entry in record:
                json.dump(entry, outfile)
                outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract audio with whisper.')
    FeatureWhisper.add_arguments(parser)
    args = parser.parse_args()
    extractor = FeatureWhisper(args)
    extractor.extract()