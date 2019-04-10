import ffmpy
from pathlib import Path
from typing import Dict, Tuple, List, Union


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
