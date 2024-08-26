import sys
sys.path.append("")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment
import subprocess
import os

from moviepy.editor import VideoFileClip

# import ffmpeg

@np.vectorize
def seconds_to_timecode(seconds):
    """ Convert seconds to timecode string (HH:MM:SS.MS) """
    seconds = np.round(seconds, 3)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def trim_audio(intervals, input_file_path, output_file_path):
        # load the audio file
    audio = AudioSegment.from_file(input_file_path)

        # iterate over the list of time intervals
    for i, (start_time, end_time) in enumerate(intervals):
            # extract the segment of the audio
        segment = audio[start_time*1000:end_time*1000]

            # construct the output file path
        output_file_path_i = f"{output_file_path}_{i}.wav"

            # export the segment to a file
        segment.export(output_file_path_i, format='wav')

def main(args):

    video_path = args.input
    output_path = args.output
    video_id = args.video_id

    scenes = pd.read_csv(args.scenes_file)

    video = VideoFileClip(video_path)

    # Extract and save the audio
    video.audio.write_audiofile("temp.wav")

    # command = "ffmpeg -i " + video_path +" -ab 160k -ac 2 -ar 16000 -vn temp.wav"
    # ffmpeg.input(video_path).output(output_path +".wav").run()
    # subprocess.call(command, shell=True)

    """Example of pyscenedetect CSV file
        Scene Number  Start Frame Start Timecode  Start Time (seconds)  End Frame  End Timecode  End Time (seconds)  Length (frames) Length (timecode)  Length (seconds)
    0               1            1   00:00:00.000                  0.00         50  00:00:02.000                2.00               50      00:00:02.000              2.00
    1               2           51   00:00:02.000                  2.00         59  00:00:02.360                2.36                9      00:00:00.360              0.36
    2               3           60   00:00:02.360                  2.36        246  00:00:09.840                9.84              187      00:00:07.480              7.48
    3               4          247   00:00:09.840                  9.84        296  00:00:11.840               11.84               50      00:00:02.000              2.00
    4               5          297   00:00:11.840                 11.84        297  00:00:11.880               11.88                1      00:00:00.040              0.04
    ..            ...          ...            ...                   ...        ...           ...                 ...              ...               ...               ...
    158           159         7201   00:04:48.000                288.00       7830  00:05:13.200              313.20              630      00:00:25.200             25.20
    159           160         7831   00:05:13.200                313.20       7900  00:05:16.000              316.00               70      00:00:02.800              2.80
    160           161         7901   00:05:16.000                316.00       8340  00:05:33.600              333.60              440      00:00:17.600             17.60
    161           162         8341   00:05:33.600                333.60       8350  00:05:34.000              334.00               10      00:00:00.400              0.40
    162           163         8351   00:05:34.000                334.00       8786  00:05:51.440              351.44              436      00:00:17.440             17.44
    """

    # fps_estimate = scenes.iloc[-1]['End Frame'] / scenes.iloc[-1]['End Time (seconds)']

    # splitted_scenes = []
    time_stamp = []
    for _, scene in scenes.iterrows():
        scene_start_sec = scene['Start Time (seconds)']
        scene_len_sec = scene['End Time (seconds)']
        time_stamp.append([scene_start_sec, scene_len_sec])




    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trim_audio(time_stamp, "temp.wav", output_path + "/" + video_id)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split long scenes in pyscenedetect CSV scene files')
    parser.add_argument('scenes_file', type=Path, help="Path to CSV file.")
    parser.add_argument('--video_id', type=str, help="Video ID.")
    parser.add_argument('--input', type=str, help="Input video.")
    parser.add_argument('--output', type=str, help="Output audio.")
    args = parser.parse_args()
    main(args)
