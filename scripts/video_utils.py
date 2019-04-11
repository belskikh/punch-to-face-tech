import ffmpy
import subprocess
import json
from pathlib import Path
from typing import Dict, Tuple, List, Union


class VideoInfo:

    def __init__(self, data: Dict) -> None:
        self.__video_info: Dict = data

    @classmethod
    def load_from_file(cls, filename: Union[str, Path]):
        filename = str(filename)
        with open(filename) as json_file:
            data = json.load(json_file)
        return cls(data)

    @classmethod
    def create_from_data(cls, data: Dict):
        return cls(data)

    def __check_data(self):
        assert self.__video_info is not None, 'No video info data was loaded!'

    def get_fps(self) -> float:
        self.__check_data()
        fps = eval(self.__video_info['streams'][0]['r_frame_rate'])
        return fps

    def get_width(self) -> int:
        self.__check_data()
        width = self.__video_info['streams'][0]['width']
        return int(width)

    def get_height(self) -> int:
        self.__check_data()
        height = self.__video_info['streams'][0]['height']
        return int(height)

    def get_nb_frames(self) -> int:
        self.__check_data()
        num_frames = self.__video_info['streams'][0]['nb_frames']
        return int(num_frames)

    def save(self, filename: Union[str, Path]) -> None:
        self.__check_data()
        filename = str(filename)
        with open(filename, 'w') as json_file:
            json.dump(self.__video_info, json_file)


def extract_frames(
        filename: Union[str, Path],
        output_dir: Union[str, Path],
        verbose: bool = True) -> None:
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # prepare source and target paths
    src_path = str(filename)
    target_path = str(output_dir/'%d.jpg')
    # input and output options
    input_opts = None
    # -b:v 10000k - average bitrate 10mb
    # -vsync 0 - all frames
    # -start_number 0 - start with frame number 0
    # -an - skip audio channels, use video only
    # -y - always overwrite
    # -q:v 2 - best quality for jpeg
    output_opts = '-start_number 0 -b:v 10000k -vsync 0 -an -y -q:v 2'
    # ffmpeg arguments
    inputs = {src_path: input_opts}
    outputs = {target_path: output_opts}
    # ffmpeg object
    ff = ffmpy.FFmpeg(inputs=inputs, outputs=outputs)
    # print cmd
    if verbose:
        print(f'ffmpeg cmd: {ff.cmd}')
    ff.run()


def get_video_info(
        filename: Union[str, Path],
        verbose: bool = True) -> Dict:
    src_path = str(filename)
    # input options
    input_opts = [
        '-v', 'quiet',
        '-print_format', 'json',
        '-select_streams', 'v:0',
        '-show_streams',
        '-show_format'
    ]
    # ffprobe arguments
    inputs = {src_path: input_opts}
    ff = ffmpy.FFprobe(inputs=inputs)
    if verbose:
        print(f'ffprobe cmd: {ff.cmd}')
    stdout, stderr = ff.run(stdout=subprocess.PIPE)
    return json.loads(stdout)
