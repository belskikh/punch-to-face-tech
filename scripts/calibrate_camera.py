import numpy as np
import cv2
from sys import float_info as flt_inf
from pathlib import Path
from typing import Dict, Tuple, List, Union
from parse_cvat_annotation import parse_cvat_xml, Annotation, Point


__all__ = [
    'parse_cvat_xml',
    'calibrate_camera',
    'OctagonMarker',
    'CameraCalibration'
]


# aliases
FramePointsMap = Dict[int, np.ndarray]
PointPairs = Tuple[np.ndarray, np.ndarray]


class OctagonMarker:

    def __init__(
            self, side: float = 1.0,
            center: Tuple[float, float] = (0.0, 0.0)) -> None:
        # длина ребра
        self.side: float = side
        # центр метки (x, y)
        self.center: Tuple[float, float] = center
        # высота метки
        self.height: float = self.side * (1 + np.sqrt(2))
        # высота ребра (по y-axis) под наклоном
        # например от pid=1 до pid=2
        self.y_diff: float = self.side / np.sqrt(2)
        # инициализируем точки нашей метки
        self._init_points()

    def _get_inverse_transform_mat(self) -> np.ndarray:
        # transform octagon center to origin
        inv_transform = np.array([
            [1.0, 0.0, -self.center[0]],
            [0.0, 1.0, -self.center[1]],
            [0.0, 0.0, 1.0]
        ]).T
        return inv_transform

    # rotation matrix for getting octagon points
    def _get_rotation_mat(self) -> np.ndarray:
        # rotation degree in radians
        angle: float = np.deg2rad(-45.0)
        # rotation matrix
        rotation = np.array([
            [np.cos(angle), np.sin(angle) * -1, 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0]
        ]).T
        # transform origin to octagon center
        transform = np.array([
            [1.0, 0.0, self.center[0]],
            [0.0, 1.0, self.center[1]],
            [0.0, 0.0, 1.0]
        ]).T
        # transform octagon center to origin
        inv_transform = self._get_inverse_transform_mat()
        # result matrix
        result = inv_transform.dot(rotation).dot(transform)
        return result

    # инициализируем точки метки
    def _init_points(self) -> None:
        # матрица поворота для получения всех точек октагона
        self._rot_mat: np.ndarray = self._get_rotation_mat()

        points = []
        # first point, id=0
        x: float = self.center[0] - self.side / 2.0
        y: float = self.center[1] + self.height / 2.0
        z: float = 1.0
        points.append(np.array([[x, y, z]]))
        # поворачиваем на 45 градусов и получаем координаты следующих точек
        for pid in range(1, 8):
            points.append(points[pid - 1].dot(self._rot_mat))
        # convert to numpy array
        self.points = np.array(points, dtype=points[0].dtype)
        # init points in 3D -> (x, y, 0)
        points_3D = self.points.copy()
        points_3D[:, :, 2] = 0.0
        self.points_3D = points_3D

    def get_points(self) -> np.ndarray:
        return self.points

    def get_points_3D(self) -> np.ndarray:
        return self.points_3D

    def _get_3D_to_img_mat(
            self,
            # width, height
            marker_size: Tuple[int, int]) -> np.ndarray:
        # transform marker from 3D to image
        w, h = marker_size
        mat = np.array([
            [w // 2, 0.0, w // 2],
            [0.0, -(h // 2), h // 2],
            [0.0, 0.0, 1.0]
        ]).T
        return mat

    def get_2D_texture_projection(
            self,
            # width, height
            img_size: Tuple[int, int],
            # width, height
            marker_size: Tuple[int, int]) -> np.ndarray:

        flag = img_size[0] >= marker_size[0]
        flag = flag and img_size[1] >= marker_size[1]
        flag = flag and img_size[0] > 0 and img_size[1] > 0
        flag = flag and marker_size[0] > 0 and marker_size[1] > 0
        flag = flag and marker_size[0] % 2 == 0
        flag = flag and marker_size[1] % 2 == 0

        assert flag, 'Invalid img size or marker size'

        # transform octagon center to origin
        inv_transform = self._get_inverse_transform_mat()
        points = self.get_points().copy().dot(inv_transform)

        # normalize points
        points = points / points.max(axis=0)

        # map marker to 2D
        # relative to marker size
        mat = self._get_3D_to_img_mat(marker_size)
        points_img = points.dot(mat)

        # from marker coordinate system
        # to image coordinate system
        img_w, img_h = img_size
        mark_w, mark_h = marker_size
        start_x = img_w // 2 - mark_w // 2
        start_y = img_h // 2 - mark_h // 2
        points[:, :, 0] += start_x
        points[:, :, 1] += start_y
        points = points[:, :, 0:2]

        # Round elements
        return np.rint(points)

    def get_max_length(self):
        return 8


class CameraCalibration:

    def __init__(self, anno: Annotation, marker: OctagonMarker) -> None:
        self.anno: Annotation = anno
        self.marker: OctagonMarker = marker
        self._prepare_points()

    # для каждой 2D точки в кадре
    # находим её 3D координаты
    def _get_frame_points(self, points: List[Point]) -> PointPairs:
        # marker planer points (x, y, 0)
        # sorted by point ID
        marker_points_in_3D: np.ndarray = self.marker.get_points_3D()

        points_in_2D = []
        points_in_3D = []

        for point in points:
            points_in_2D.append([point.x, point.y])
            points_in_3D.append(marker_points_in_3D[point.id].copy())

        points_in_2D = np.array(points_in_2D, dtype=np.float32).reshape(-1, 1, 2)
        points_in_3D = np.array(points_in_3D, dtype=np.float32).reshape(-1, 1, 3)

        return points_in_2D, points_in_3D

    # для каждой 2D точки
    # находим её 3D координаты
    # для всех кадров с маркером
    def _prepare_points(self) -> None:
        # 2D points in image plane
        self.img_points = []
        # 3D points in real world space
        self.obj_points = []
        # frame order
        self.frame_order = []

        for frame_n, points in self.anno.get_frames().items():
            points_in_2D, points_in_3D = self._get_frame_points(points)
            self.img_points.append(points_in_2D)
            self.obj_points.append(points_in_3D)
            self.frame_order.append(frame_n)

    # calibrate camera
    # default flags:
    # cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
    def calibrate(
            self, flags: int = 6,
            criteria: Tuple[int, int, float] = (3, 30, flt_inf.epsilon)
        ):
        # ret, mtx, dist, rvecs, tvecs
        return cv2.calibrateCamera(
            self.obj_points, self.img_points,
            (self.anno.width, self.anno.height),
            None, None,
            flags=flags,
            criteria=criteria
        )

# calibrate camera
# default flags:
# cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
def calibrate_camera(
        anno_fn: Union[str, Path], flags: int = 6,
        criteria: Tuple[int, int, float] = (3, 30, flt_inf.epsilon)
    ):
    anno = parse_cvat_xml(anno_fn)
    marker = OctagonMarker()

    camera_calib = CameraCalibration(anno, marker)

    return camera_calib.calibrate(flags=flags, criteria=criteria)
