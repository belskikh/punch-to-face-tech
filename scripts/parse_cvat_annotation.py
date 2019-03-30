from lxml import etree
from typing import Union, Tuple
from pathlib import Path
from collections import defaultdict


__all__ = ['parse_cvat_xml']


class Point:

    def __init__(self, point_id:int, coords:Tuple[float, float]):
        self.id = point_id
        self.x = coords[0]
        self.y = coords[1]


class Annotation:

    def __init__(self, filename:Path, width:int, height:int):
        self.filename = filename
        self.name = filename.stem
        self.width = width
        self.height = height
        self.frames = defaultdict(list)

    def add_point(self, frame_n:int, point:Point) -> None:
        self.frames[frame_n].append(point)

    # должно быть минимум 4 точки в каждом кадре
    def _validate(self) -> None:
        frames = {}
        for frame_n, points in self.frames.items():
            if len(points) >= 4:
                frames[frame_n] = points
        self.frames = frames


def parse_cvat_xml(filename:Union[str,Path]) -> Annotation:
    filename = Path(filename)
    root = etree.parse(str(filename)).getroot()

    # размер видео
    width = int(root.find(".//meta/task/original_size/width").text)
    height = int(root.find(".//meta/task/original_size/height").text)

    # результат
    anno = Annotation(filename=filename, width=width, height=height)

    for track_tag in root.findall(".//track[@label='marker']"):
        point_tag = track_tag.find(".//points[@outside='0']")
        # номер кадра
        frame_n = int(point_tag.attrib['frame'])
        # ID точки
        point_id = int(point_tag.find(".//attribute[@name='id']").text)
        # если по ошибке добавили несколько точек
        # то складываем только первую точку
        coords = point_tag.attrib['points'].split(';')[0].split(',')
        # координаты отдельной точки маркера
        coords = tuple(map(float, coords))
        # добавляем точку
        point = Point(point_id, coords)
        anno.add_point(frame_n, point)

    # валидируем
    # должно быть минимум 4 точки в каждом кадре
    anno._validate()
    return anno
