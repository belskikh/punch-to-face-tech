import numpy as np
from typing import Dict, Tuple, List
from parse_cvat_annotation import parse_cvat_xml, Annotation, Point


# __all__ = ['parse_cvat_xml']


# aliases
FramePoints = Dict[int, np.ndarray]
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

    # матрица поворота, для получения точек октагона
    def _get_rotation_mat(self) -> np.ndarray:
        # угол поворота в радианах
        angle: float = np.deg2rad(-45.0)
        return np.array([
            [np.cos(angle), np.sin(angle) * -1],
            [np.sin(angle), np.cos(angle)]
        ]).T

    # инициализируем точки метки
    def _init_points(self) -> None:
        # матрица поворота для получения всех точек октагона
        self._rot_mat: np.ndarray = self._get_rotation_mat()

        self.points = {}
        # first point, id=0
        x: float = self.center[0] - self.side / 2.0
        y: float = self.center[1] + self.height / 2.0
        self.points[0] = np.array([[x, y]])
        # поворачиваем на 45 градусов и получаем координаты следующих точек
        for pid in range(1, 8):
            self.points[pid] = self.points[pid-1].dot(self._rot_mat)

    def get_points(self) -> FramePoints:
        return self.points

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
        #  pid -> 3D координата на плоскости без Z (x, y)
        marker_points_in_3D: FramePoints = self.marker.get_points()

        points_in_2D = []
        points_in_3D = []

        for point in points:
            points_in_2D.append([point.x, point.y])
            points_in_3D.append(marker_points_in_3D[point.id])

        points_in_2D = np.array(points_in_2D, dtype=np.float32).reshape(-1, 2)
        points_in_3D = np.array(points_in_3D, dtype=np.float32).reshape(-1, 2)

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
    def calibrate(self, flags: int = 6):
        # ret, mtx, dist, rvecs, tvecs
        return cv2.calibrateCamera(
            self.obj_points, self.img_points,
            (self.anno.width, self.anno.height),
            None, None,
            flags=flags
        )
