import numpy as np
# cv2.__version__ == '3.4.2'
import cv2

from pathlib import Path
from typing import Tuple, List, Union, Dict
from tqdm import tqdm

from video_utils import VideoScene


__all__ = [
    'SIFT',
    'calc_scene_homography'
]


# aliases
Matches = List[Tuple[int, int]]
HomographyResult = Tuple[Matches, np.ndarray, np.ndarray]


class SIFT:

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

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return matches, H, status

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(
            self,
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
        vis[0:h2, w1:] = image1

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
            masks: Tuple[np.ndarray, np.ndarray]) -> HomographyResult:

        image1, image2 = images
        mask1, mask2 = masks

        kps1, features1 = self.detect_and_describe(image1, mask1)
        kps2, features2 = self.detect_and_describe(image2, mask2)

        # # match features between the two images
        M = self.match_keypoints(kps1, kps2, features1, features2)

        return M


def calc_scene_homography(
        video_scene: VideoScene,
        frame_dir: Union[str, Path],
        mask_dir: Union[str, Path]) -> Dict[int, HomographyResult]:

    sift = SIFT()

    prev_frame = None
    prev_mask = None
    result = {}

    first_frame = video_scene.get_first_frame()
    num_frames = video_scene.get_num_frames()

    for frame_n in tqdm(range(first_frame + 1, num_frames)):
        prev_frame_n = frame_n - 1
        if prev_frame is None:
            frame_path = frame_dir / f'{prev_frame_n}.jpg'
            mask_path = mask_dir / f'{prev_frame_n}.png'
            prev_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            prev_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        frame_path = frame_dir / f'{frame_n}.jpg'
        mask_path = mask_dir / f'{frame_n}.png'
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        images = (prev_frame, frame)
        masks = (prev_mask, mask)
        M = sift.calc_homography(images, masks)

        result[prev_frame_n] = M

        prev_frame = frame
        prev_mask = mask

    return result
