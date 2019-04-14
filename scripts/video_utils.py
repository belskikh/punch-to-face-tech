import ffmpy
import subprocess
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Union
from utils import IMG_EXT


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


class VideoScene:

    def __init__(self, id: int, first_frame: int, last_frame: int) -> None:
        self.id = id
        self.first_frame = first_frame
        self.last_frame = last_frame

    def get_id(self) -> int:
        return self.id

    def get_first_frame(self) -> int:
        return self.first_frame

    def get_last_frame(self) -> int:
        return self.last_frame

    def get_num_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    def contains(self, frame_n: int):
        flag = frame_n >= self.first_frame
        flag = flag and frame_n <= self.last_frame
        return flag


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
        # init video scenes
        self.scenes = self.__init_scenes()

        # save important video attributes
        self.width = self.frames[0].get_width()
        self.height = self.frames[0].get_height()
        self.num_frames = len(self.frames)
        self.r_frame_rate = self.__video_fps_info['r_frame_rate']
        self.avg_frame_rate = self.__video_fps_info['avg_frame_rate']
        self.time_base = self.__video_fps_info['time_base']
        self.fps = eval(self.r_frame_rate)

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
        result = sorted(list(map(VideoInfo.__ts_map_func, scene_frame_ts_arr)))
        # add first frame pts
        result.insert(0, self.get_frames()[0].get_pkt_pts())
        return result

    def __init_scenes(self) -> List[VideoScene]:
        # map pts to frames
        pts_to_frames = {}
        for frame_n, frame in enumerate(self.get_frames()):
            pts_to_frames[frame.get_pkt_pts()] = frame_n
        # array of video scenes
        video_scenes = []
        scene_id = 0
        scene_len = len(self.get_scene_frame_ts())
        prev_video_scene = None
        # iterate over scene pts
        for idx, pts in enumerate(self.get_scene_frame_ts()):
            first_frame = pts_to_frames[pts]
            if idx == (scene_len - 1):
                last_frame = len(self.frames) - 1
            else:
                last_frame_pts = self.get_scene_frame_ts()[idx + 1]
                last_frame = pts_to_frames[last_frame_pts] - 1
            # if it was a one frame scene before
            if prev_video_scene is not None:
                video_scene.last_frame = last_frame
            else:
                video_scene = VideoScene(scene_id, first_frame, last_frame)
                video_scenes.append(video_scene)
                scene_id += 1
            # check if it's one frame scene
            frame_diff = last_frame - first_frame
            if frame_diff == 0:
                prev_video_scene = video_scene
            else:
                prev_video_scene = None
        return video_scenes

    def show_scenes_info(self) -> None:
        print()
        print(f'{self.name} scenes:')
        for scene in self.get_scenes():
            scene_id = scene.get_id()
            first_frame = scene.get_first_frame()
            last_frame = scene.get_last_frame()
            info = f'\t{scene_id: >3} - ({first_frame: >6}, {last_frame: >6})'
            print(info)
        print()

    def save(
            self,
            output_dir: Union[str, Path] = '../data/video/info') -> None:
        # create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # result filename
        filename = output_dir/f'{self.name}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(
            name: str,
            directory: Union[str, Path] = '../data/video/info') -> 'VideoInfo':
        filename = Path(directory)/f'{name}.pickle'
        video_info = None
        with open(filename, 'rb') as f:
            video_info = pickle.load(f)
        return video_info

    @classmethod
    def create_video_info(
            cls,
            filename: Union[str, Path],
            scene_thresh: float = 0.2,
            verbose: bool = True) -> 'VideoInfo':
        frame_arr = get_frames_info(filename)
        scene_frame_ts_arr = get_scene_frame_ts(filename, scene_thresh)
        video_fps_info = get_video_fps_info(filename)
        video_name = Path(filename).stem
        return cls(video_name, frame_arr, scene_frame_ts_arr, video_fps_info)

    def get_name(self) -> str:
        return self.name

    def get_frames(self) -> List[VideoFrame]:
        return self.frames

    def get_scene_frame_ts(self) -> List[int]:
        return self.scene_frame_ts

    def get_scenes(self) -> List[VideoScene]:
        return self.scenes

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_num_frames(self) -> int:
        return self.num_frames

    def get_fps(self) -> float:
        return self.fps


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


# frames have to be in frame_dir already
# after extract_frames(...) method
def save_scenes_first_last_frames(
        scenes: List[VideoScene],
        frame_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extension: str = '.jpg',
        ignore_errors: bool = False,
        verbose: bool = True) -> None:
    frame_dir = Path(frame_dir)
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # frames to save
    scene_frames: List[Tuple[int, int]] = []
    for scene in scenes:
        scene_id = scene.get_id()
        scene_frames.append((scene_id, scene.get_first_frame()))
        scene_frames.append((scene_id, scene.get_last_frame()))
    # iterate over frames
    for scene_frame in scene_frames:
        scene_id = scene_frame[0]
        frame_n = scene_frame[1]
        src_path = frame_dir/f'{frame_n}{extension}'
        if not src_path.exists():
            if ignore_errors:
                print(f'{src_path} doesn\'t exist')
                continue
            else:
                raise FileNotFoundError(src_path)
        dst_path = output_dir/f'{scene_id}_{frame_n}{extension}'
        shutil.copy(str(src_path), str(dst_path))
        if verbose:
            print(f'done with {dst_path.name}')


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
