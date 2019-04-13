import ffmpy
import subprocess
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Union


class VideoFrame:

    def __init__(self, frame: Dict) -> None:
        self.__data = frame
        self.pkt_pts = int(frame['pkt_pts'])
        self.pkt_pts_time = float(frame['pkt_pts_time'])
        self.pkt_duration = int(frame['pkt_duration'])
        self.pkt_duration_time = float(frame['pkt_duration_time'])
        self.width = int(frame['width'])
        self.height = int(frame['height'])

    def get_pkt_pts(self) -> int:
        return self.pkt_pts

    def get_pkt_pts_time(self) -> float:
        return self.pkt_pts_time

    def get_pkt_duration(self) -> int:
        return self.pkt_duration

    def get_pkt_duration_time(self) -> float:
        return self.pkt_duration_time

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_data(self) -> Dict:
        return self.__data


class VideoInfo:

    def __init__(
            self,
            name: str,
            frame_arr: List[Dict],
            scene_frame_ts_arr: List[Dict[str, int]],
            video_fps_info: Dict) -> None:

        # name of the video
        self.name = name

        # save original data
        self.__frame_arr = frame_arr
        self.__scene_frame_ts_arr = scene_frame_ts_arr
        self.__video_fps_info = video_fps_info

        # convert json frames to VideoFrame
        self.frames = self.__init_frames(self.__frame_arr)
        # convert json frame timestamps to List[int]
        self.scene_frame_ts = self.__init_scene_frame_ts(self.__scene_frame_ts_arr)

        # save important video attributes
        self.width = self.frames[0].get_width()
        self.height = self.frames[0].get_height()
        self.num_frames = len(self.frames)
        self.r_frame_rate = self.__video_fps_info['r_frame_rate']
        self.avg_frame_rate = self.__video_fps_info['avg_frame_rate']
        self.time_base = self.__video_fps_info['time_base']
        self.fps = eval(self.r_frame_rate)

        # get scene pairs (first_frame_in_scene, last_frame_in_scene)

    @staticmethod
    def __frame_sort_func(frame: VideoFrame) -> int:
        return frame.get_pkt_pts()

    @staticmethod
    def __ts_map_func(ts_dict: Dict[str, int]) -> int:
        return ts_dict['pkt_pts']

    def __init_frames(self, frame_arr: List[Dict]) -> List[VideoFrame]:
        sort_fn = VideoInfo.__frame_sort_func
        return sorted(list(map(VideoFrame, frame_arr)), key=sort_fn)

    def __init_scene_frame_ts(
            self,
            scene_frame_ts_arr: List[Dict[str, int]]) -> List[int]:
        return sorted(list(map(VideoInfo.__ts_map_func, scene_frame_ts_arr)))

    def get_name(self) -> str:
        return self.name

    def get_frames(self) -> List[VideoFrame]:
        return self.frames

    def get_scene_frame_ts(self) -> List[int]:
        return self.scene_frame_ts

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_num_frames(self) -> int:
        return self.num_frames

    def get_fps(self) -> float:
        return self.fps

    def save(self, output_dir: Union[str, Path]) -> None:
        filename = Path(output_dir)/f'{self.name}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: Union[str, Path]) -> VideoInfo:
        video_info = None
        with open(filename, 'wb') as f:
            video_info = pickle.load(f)
        return video_info


    # или лучше по номеру кадра их как-то сделать????






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
        scene_thresh: float = 0.2,
        verbose: bool = True) -> VideoInfo:
    frame_arr = get_frames_info(filename)
    scene_frame_ts_arr = get_scene_frame_ts(filename, scene_thresh)
    video_fps_info = get_video_fps_info(filename)
    video_name = Path(filename).stem
    return VideoInfo(video_name, frame_arr, scene_frame_ts_arr, video_fps_info)


def probe_video(
        filename: Union[str, Path],
        input_opts: List[str],
        verbose: bool = True) -> Tuple[bytes, bytes]:
    src_path = str(filename)
    if not verbose:
        input_opts.append('-v')
        input_opts.append('quiet')
    # ffprobe arguments
    inputs = {src_path: input_opts}
    ff = ffmpy.FFprobe(inputs=inputs)
    if verbose:
        print(f'ffprobe cmd: {ff.cmd}')
    stdout, stderr = ff.run(stdout=subprocess.PIPE)
    return stdout, stderr


def get_frames_info(
        filename: Union[str, Path],
        verbose: bool = True) -> List[Dict]:
    # input options
    input_opts = [
        '-print_format', 'json',
        '-select_streams', 'v:0',
        '-show_frames',
        '-show_entries',
        'frame=pkt_pts,pkt_pts_time,pkt_duration,pkt_duration_time,width,height'
    ]
    stdout, stderr = probe_video(filename, input_opts, verbose)
    return json.loads(stdout)['frames']


def get_scene_frame_ts(
        filename: Union[str, Path],
        scene_thresh: float = 0.2,
        verbose: bool = True) -> List[Dict[str, int]]:
    src_path = f'movie={filename}, select=gt(scene\\,{scene_thresh})'
    # input options
    input_opts = [
        '-print_format', 'json',
        '-select_streams', 'v:0',
        '-f', 'lavfi',
        '-show_frames',
        '-show_entries',
        'frame=pkt_pts'
    ]
    stdout, stderr = probe_video(src_path, input_opts, verbose)
    return json.loads(stdout)['frames']


def get_video_fps_info(
        filename: Union[str, Path],
        verbose: bool = True) -> Dict:
    # input options
    input_opts = [
        '-print_format', 'json',
        '-select_streams', 'v:0',
        '-show_streams',
        '-show_entries',
        'stream=r_frame_rate,avg_frame_rate,time_base'
    ]
    stdout, stderr = probe_video(filename, input_opts, verbose)
    return json.loads(stdout)['streams'][0]
