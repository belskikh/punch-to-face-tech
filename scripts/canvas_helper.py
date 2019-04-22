import numpy as np
# cv2.__version__ == '3.4.2'
import cv2

from pathlib import Path
from typing import Tuple, List, Union, Dict
from tqdm import tqdm
import json

from video_utils import VideoScene
from calibrate_camera import OctagonMarker
from parse_cvat_annotation import Point
import utils


# aliases
Matches = List[Tuple[int, int]]
HomographyResult = Tuple[Matches, np.ndarray, np.ndarray]


class HomographyHelper:

    def __init__(
            self,
            ratio: float=0.75,
            reproj_thresh: float=4.0) -> None:
        # cv2.__version__ == '3.4.2'
        # detect and extract features from the image
        self.descriptor = cv2.xfeatures2d.SIFT_create()
        # keypoints matcher
        self.matcher = cv2.BFMatcher()
        # for matching
        self.ratio = ratio
        # for findHomography
        self.reproj_thresh = reproj_thresh

    def detect_and_describe(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            draw_keypoints: bool = False
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect and extract features from the image
        kps, features = self.descriptor.detectAndCompute(
            image=gray,
            mask=mask
        )

        if draw_keypoints:
            vis = np.zeros_like(image)
            cv2.drawKeypoints(
                image=gray,
                keypoints=kps,
                outImage=vis,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # return a tuple of keypoints, features and visualization
            return kps, features, vis

        # return a tuple of keypoints and features
        return kps, features

    def match_keypoints(
            self,
            kps1: List[cv2.KeyPoint],
            kps2: List[cv2.KeyPoint],
            features1: np.ndarray,
            features2: np.ndarray) -> HomographyResult:

        # convert the keypoints from KeyPoint objects
        # to NumPy arrays
        kps1 = np.float32([kp.pt for kp in kps1])
        kps2 = np.float32([kp.pt for kp in kps2])

        # compute the raw matches and initialize
        # the list of actual matches
        raw_matches = self.matcher.knnMatch(features1, features2, k=2)
        matches = []

        # loop over the raw matches
        for match in raw_matches:
            if len(match) != 2:
                continue
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            match_flag = match[0].distance < match[1].distance * self.ratio
            if match_flag:
                matches.append((match[0].trainIdx, match[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            pts1 = np.float32([kps1[i] for (_, i) in matches])
            pts2 = np.float32([kps2[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            H, status = cv2.findHomography(
                srcPoints=pts1,
                dstPoints=pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.reproj_thresh
            )

            # return the matches along with the homography matrix
            # and status of each matched point
            return matches, H, status

        # otherwise, no homography could be computed
        return None

    @staticmethod
    def draw_matches(
            image1: np.ndarray,
            image2: np.ndarray,
            kps1: List[cv2.KeyPoint],
            kps2: List[cv2.KeyPoint],
            matches: Matches,
            status: np.ndarray):

        # convert the keypoints from KeyPoint objects
        # to NumPy arrays
        kps1 = np.float32([kp.pt for kp in kps1])
        kps2 = np.float32([kp.pt for kp in kps2])

        # initialize the output visualization image
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
        vis[0:h1, 0:w1] = image1
        vis[0:h2, w1:] = image2

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                pt1 = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
                pt2 = (int(kps2[trainIdx][0]) + w1, int(kps2[trainIdx][1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

        # return the visualization
        return vis

    def calc_homography(
            self,
            images: Tuple[np.ndarray, np.ndarray],
            masks: Tuple[np.ndarray, np.ndarray],
            draw_matches: bool = False
    ) -> Union[HomographyResult, Tuple[HomographyResult, np.ndarray]]:

        image1, image2 = images
        mask1, mask2 = masks

        kps1, features1 = self.detect_and_describe(image1, mask1)
        kps2, features2 = self.detect_and_describe(image2, mask2)

        # # match features between the two images
        M = self.match_keypoints(kps1, kps2, features1, features2)

        if M is not None and draw_matches:
            vis = HomographyHelper.draw_matches(
                image1=image1,
                image2=image2,
                kps1=kps1,
                kps2=kps2,
                matches=M[0],
                status=M[2]
            )
            return M, vis

        return M


def get_point_pairs_2D(
        frame_points: List[Point],
        texture_points: np.ndarray):

    points_in_img = []
    points_in_texture = []

    for point in frame_points:
        points_in_img.append(point.get_coords())
        points_in_texture.append(texture_points[point.get_id()].copy())

    points_in_img = np.array(
        points_in_img,
        dtype=np.float32).reshape(-1, 1, 2)
    points_in_texture = np.array(
        points_in_texture,
        dtype=np.float32).reshape(-1, 1, 2)

    return points_in_img, points_in_texture


def get_img_texture_homography(
        frame_points: List[Point],
        marker: OctagonMarker,
        texture_size: Tuple[int, int] = (1024, 1024),
        marker_size: Tuple[int, int] = (600, 600)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    texture_points = marker.get_2D_texture_projection(
        img_size=texture_size,
        marker_size=marker_size
    )

    points_in_img, points_in_texture = get_point_pairs_2D(
        frame_points=frame_points,
        texture_points=texture_points
    )

    H, status = cv2.findHomography(points_in_img, points_in_texture)
    return H, status, texture_points


def warp_img_to_texture(
        img: np.ndarray,
        mask: np.ndarray,
        homography: np.ndarray,
        texture_size: Tuple[int, int] = (1024, 1024)
) -> Tuple[np.ndarray, np.ndarray]:

    img_warped = cv2.warpPerspective(
        src=img, M=homography,
        dsize=texture_size,
        flags=cv2.INTER_LANCZOS4
    )

    mask_warped = cv2.warpPerspective(
        src=mask, M=homography,
        dsize=texture_size,
        flags=cv2.INTER_NEAREST
    )
    return img_warped, mask_warped


def calc_homography_and_warp(
        img: np.ndarray,
        mask: np.ndarray,
        frame_points: List[Point],
        marker: OctagonMarker,
        texture_size: Tuple[int, int] = (1024, 1024),
        marker_size: Tuple[int, int] = (600, 600)
) -> Tuple[np.ndarray, np.ndarray]:

    H, status, texture_points = get_img_texture_homography(
        frame_points=frame_points,
        marker=marker,
        texture_size=texture_size,
        marker_size=marker_size
    )
    img_warped, mask_warped = warp_img_to_texture(
        img=img,
        mask=mask,
        homography=H,
        texture_size=texture_size
    )
    return img_warped, mask_warped


def calc_scene_homography(
        init_homo: np.ndarray,
        video_scene: VideoScene,
        points0: np.ndarray,
        point0_ids: List[int],
        texture_points: np.ndarray,
        frame_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        texture_dir: Union[str, Path],
        texture_size: Tuple[int, int] = (1280, 1280),
        result_name: str = (
            'ufc234_gastelum_bisping_1080p_nosound_cut'
            '__homography'
        ),
        result_dir: Union[str, Path] = '../data/video/info',
        draw_points: bool = False,
        point_color: Tuple[int, int, int] = (153, 255, 153),
        point_radius: int = 10,
        point_font_scale: float = 1.5,
        point_font_thickness: int = 2
) -> Dict[int, Dict[str, np.ndarray]]:

    frame_dir = Path(frame_dir)
    mask_dir = Path(mask_dir)

    # result directories
    result_dir = Path(result_dir)
    texture_dir = Path(texture_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[int, Dict[str, np.ndarray]] = {}

    homo = HomographyHelper()

    first_frame_n = video_scene.get_first_frame()
    num_frames = video_scene.get_num_frames()

    # save init_homo to result
    prev_frame_n = first_frame_n
    result[prev_frame_n] = {'texture': init_homo}

    # save previous frame and mask
    prev_frame, prev_mask = _load_frame_and_mask(
        frame_n=prev_frame_n,
        frame_dir=frame_dir,
        mask_dir=mask_dir
    )

    # init texture
    prev_texture, prev_texture_mask = warp_img_to_texture(
        prev_frame, prev_mask, init_homo,
        texture_size=texture_size
    )

    # also save texture with image points
    if draw_points:
        img_points = utils.draw_points(
            prev_texture, texture_points,
            color=point_color,
            draw_ids=True,
            point_ids=list(range(0, 8)),
            radius=point_radius,
            font_scale=point_font_scale,
            font_thickness=point_font_thickness
        )
    else:
        img_points = None

    _save_frame_and_mask(
        frame=prev_texture,
        mask=prev_texture_mask,
        frame_n=prev_frame_n,
        output_dir=texture_dir,
        points=img_points
    )

    # we don't use frame_0
    # because all information is in init_homo
    first_frame_n += 1

    for frame_n in tqdm(range(first_frame_n, num_frames)):
        frame, mask = _load_frame_and_mask(
            frame_n=frame_n,
            frame_dir=frame_dir,
            mask_dir=mask_dir
        )
        # считаем homography между предыдущим кадром
        # в сцене и текущим кадром
        homo_res = homo.calc_homography(
            images=(prev_frame, frame),
            masks=(prev_mask, mask)
        )
        if homo_res is None:
            result[frame_n]['texture'] = None
            print(f'No homography for frame {frame_n}')
            continue
        else:
            matches, H_frames, status = homo_res

        # считаем временную homography
        # для преобразования из текущего кадра в текстуру
        homo_texture = result[prev_frame_n]['texture']
        H_tmp = homo_texture.dot(np.linalg.inv(H_frames))

        # делаем warp текущего кадра в текстуру
        tmp_texture, tmp_texture_mask = warp_img_to_texture(
            frame, mask, H_tmp,
            texture_size=texture_size
        )

        # считаем homography между текстурой
        # текущего кадра и текстурой предыдущего
        # для корректировки homography
        homo_res = homo.calc_homography(
            images=(prev_texture, tmp_texture),
            masks=(prev_texture_mask, tmp_texture_mask)
        )
        if homo_res is None:
            result[frame_n]['texture'] = None
            print(f'No homography for frame {frame_n}')
            continue
        else:
            matches, H_tex, status = homo_res

        # H_tmp = texture -> frame
        # H_tex = texture -> prev_texture
        # np.linalg.inv(H_tex) = prev_texture -> texture

        # prev_texture -> frame
        # корректируем homography
        H_new = np.linalg.inv(H_tex).dot(H_tmp)
        result[frame_n] = {'texture': H_new}

        # get textures for current frame
        texture, texture_mask = warp_img_to_texture(
            frame, mask, H_new,
            texture_size=texture_size
        )

        # also save texture with image points
        if draw_points:
            img_points = utils.draw_points(
                texture, texture_points,
                color=point_color,
                draw_ids=True,
                point_ids=list(range(0, 8)),
                radius=point_radius,
                font_scale=point_font_scale,
                font_thickness=point_font_thickness
            )
        else:
            img_points = None

        _save_frame_and_mask(
            frame=texture,
            mask=texture_mask,
            frame_n=frame_n,
            output_dir=texture_dir,
            points=img_points
        )

        # сохраняем данные с текущего кадра
        prev_frame_n = frame_n
        prev_frame, prev_mask = frame, mask
        prev_texture, prev_texture_mask = texture, texture_mask

    # save homographies
    _save_scene_homo(result, name=result_name, output_dir=result_dir)
    return result


def _load_frame_and_mask(
        frame_n: int,
        frame_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        frame_ext: str = '.jpg',
        mask_ext: str = '.png'):

    frame_path = frame_dir / f'{frame_n}{frame_ext}'
    mask_path = mask_dir / f'{frame_n}{mask_ext}'

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    return frame, mask


def _save_img(
        img: np.ndarray,
        frame_n: int,
        output_dir: Union[str, Path],
        img_ext: str = '.jpg'
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_path = output_dir / f'{frame_n}{img_ext}'
    ret = cv2.imwrite(str(img_path), img)


def _save_frame_and_mask(
        frame: np.ndarray,
        mask: np.ndarray,
        frame_n: int,
        output_dir: Union[str, Path],
        points: np.ndarray = None,
        frame_dir: Union[str, Path] = 'frames',
        mask_dir: Union[str, Path] = 'masks',
        point_dir: Union[str, Path] = 'points',
        frame_ext: str = '.jpg',
        mask_ext: str = '.png'
) -> None:

    output_dir = Path(output_dir)
    frame_dir = output_dir / frame_dir
    mask_dir = output_dir / mask_dir

    _save_img(frame, frame_n, frame_dir, img_ext=frame_ext)
    _save_img(mask, frame_n, mask_dir, img_ext=mask_ext)

    if points is not None:
        point_dir = output_dir / point_dir
        _save_img(points, frame_n, point_dir, img_ext=frame_ext)


def _save_scene_homo(
        data: Dict,
        name: str = 'ufc234_gastelum_bisping_1080p_nosound_cut__homography',
        output_dir: Union[str, Path] = '../data/video/info') -> None:
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # result filename
    filename = output_dir / f'{name}.json'
    # save
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


def load_scene_homo(
        name: str = 'ufc234_gastelum_bisping_1080p_nosound_cut__homography',
        directory: Union[str, Path] = '../data/video/info'
) -> Dict[int, Dict[str, np.ndarray]]:

        filename = Path(directory) / f'{name}.json'

        data = None
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
