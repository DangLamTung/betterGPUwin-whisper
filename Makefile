install:
	sudo apt update && sudo apt install ffmpeg && sudo apt install -y mkvtoolnix && rm -rf /var/lib/apt/lists/*
	pip install -r setup.txt

init:
# change to the folder name you want to create
	python src/init.py

import:
	python src/import/import.py --bulk --no-resize

analyze: 
	python src/analyze/analyze.py --id "visione_guide"
# python src/analyze/analyze.py --id 02082013

make dino: #test object detect ground dino
# python src/analyze/objects-ground-dino/extract.py --gpu test-collection/selected-frames/02082013 jsonl --output test-collection/objects-ground-dino/{video_id}/{video_id}-objects-ground-dino.jsonl.gz
	python src/analyze/objects-ground-dino/extract.py --gpu "test-collection/selected-frames/Phú Quốc trip" jsonl --output test-collection/objects-ground-dino/{video_id}/{video_id}-objects-ground-dino.jsonl.gz

make clip:
	python src/analyze/features-clip/extract.py --gpu --batch-size 1 'test-collection/selected-frames/02082013' 'hdf5' --features-name 'clip' --output test-collection/features-clip/{video_id}/{video_id}-clip.hdf5

make cluster:
	python src/analyze/frame-cluster/cluster.py 'test-collection/features-clip/02082013/02082013-clip.hdf5' 'test-collection/cluster-codes/02082013/02082013-cluster-codes.jsonl.gz'

ffmpeg:
# on Ubuntu or Debian
	sudo apt update && sudo apt install ffmpeg
