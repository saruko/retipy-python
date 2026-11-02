# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified 2026-11-02 for retipy-python_fork: the curve measures follow Hart 1999 and Grisan 2008
# and are computed on a smoothing spline. See README_fork.md for the list of changes.

"""Module with operations related to extracting tortuosity measures.

蛇行度は、順に並べた中心線の点に平滑化スプラインを当てて計算する。Hart 1999 (p.245) は平滑化してから計算し、
Grisan 2003, 2008 も平滑化スプラインを用いる。画素の階段をそのまま微分すると、曲率が階段の雑音で埋もれる。
"""

import numpy as np
from lib import fractal_dimension
from retipy.retina import Retina, Window, detect_vessel_border
from scipy.interpolate import splev, splprep

# 合成画像だけで決めた値（README.md 参照）
SMOOTHING = 0.35       # 点あたりの残差の二乗和の許容量（画素²）。x, y で共通なので向きによらない
HYSTERESIS = 0.01      # 曲率の符号の変化とみなすしきい値（1/画素）。Grisan 2003 の静脈の値
SAMPLES_PER_PIXEL = 2  # スプラインの標本の密度

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # NumPy 2 で np.trapz は np.trapezoid になった


def _distance_2p(x1, y1, x2, y2):
    """
    calculates the distance between two given points
    :param x1: starting x value
    :param y1: starting y value
    :param x2: ending x value
    :param y2: ending y value
    :return: the distance between [x1, y1] -> [x2, y2]
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _curve_length(x, y):
    """
    calculates the length(distance) of the given curve, iterating from point to point.
    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the curve length
    """
    distance = 0
    for i in range(0, len(x) - 1):
        distance += _distance_2p(x[i], y[i], x[i + 1], y[i + 1])
    return distance


def _chord_length(x, y):
    """
    distance between starting and end point of the given curve

    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the chord length of the given curve
    """
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])


class Centerline:
    """順に並べた点に平滑化スプラインを当てた曲線。s は弧長、kappa は符号つきの曲率（Hart 1999 の式(2)）"""

    def __init__(self, x, y):
        pts = np.column_stack([x, y]).astype(float)
        keep = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
        pts = pts[keep]
        n = len(pts)
        if n <= 3:
            raise ValueError("スプラインには4点以上が必要")
        tck, _ = splprep([pts[:, 0], pts[:, 1]], k=3, s=SMOOTHING * n)
        chain = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        t = np.linspace(0, 1, max(int(round(chain * SAMPLES_PER_PIXEL)), 8))
        self.xy = np.column_stack(splev(t, tck))
        d1 = np.column_stack(splev(t, tck, der=1))
        d2 = np.column_stack(splev(t, tck, der=2))
        speed = np.hypot(d1[:, 0], d1[:, 1])
        self.kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-9) ** 3
        steps = np.hypot(*np.diff(self.xy, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def length(self):
        return float(self.s[-1])

    @property
    def chord(self):
        return float(np.hypot(*(self.xy[-1] - self.xy[0])))


def _centerline(x, y):
    """スプラインを当てられる点の数があれば Centerline を、なければ None を返す"""
    return Centerline(x, y) if len(x) > 3 else None


def _turn_curve_cuts(line: Centerline):
    """曲率の符号が変わる標本の位置を返す。しきい値を超えた区間の間で符号が変わったら、その中点で分ける（Grisan 2008）"""
    state = np.where(line.kappa > HYSTERESIS, 1, np.where(line.kappa < -HYSTERESIS, -1, 0))
    active = np.flatnonzero(state)
    return [(a + b) // 2 for a, b in zip(active[:-1], active[1:]) if state[a] != state[b]]


def _detect_inflection_points(x, y):
    """
    曲率の符号が変わる点（変曲点）を検出する。Grisan 2008 の式(17)と同じく曲率の符号で区切る。

    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the array position in x of the inflection points.
    """
    line = _centerline(x, y)
    if line is None:
        return []
    pts = np.column_stack([x, y]).astype(float)
    return [int(np.argmin(np.hypot(*(pts - line.xy[c]).T))) for c in _turn_curve_cuts(line)]


def _curve_to_image(x, y):
    """曲線を収める正方形の二値画像を返す。入力の x, y は書き換えない"""
    rows = np.round(np.asarray(x)).astype(np.int64)
    cols = np.round(np.asarray(y)).astype(np.int64)
    rows = rows - rows.min()
    cols = cols - cols.min()
    # calculate which square with side 2^n of size will contain the line
    image_dim = 2
    while image_dim < rows.max() or image_dim < cols.max():
        image_dim *= 2
    image_dim *= 2

    image_curve = np.full([image_dim, image_dim], False)
    image_curve[rows, cols] = True

    return Retina(image_curve, "curve_image")


def linear_regression_tortuosity(x, y, sampling_size=6, retry=True):
    """
    This method calculates a tortuosity measure by estimating a line that start and ends with the
    first and last points of the given curve, then samples a number of pixels from the given line
    and calculates its determination coefficient, if this value is closer to 1, then the given
    curve is similar to a line.

    決定係数は 1 − 残差平方和 ÷ 全平方和 で求める。1 を超えることはなく、平均より当てはまりが悪ければ負になる。

    This method assumes that the given parameter is a sorted list.

    Returns the determination coefficient for the given curve
    :param x: the x component of the curve
    :param y: the y component of the curve
    :param sampling_size: how many pixels
    :param retry: if regression fails due to a zero division, try again by inverting x and y
    :return: the coefficient of determination of the curve.
    """
    if len(x) < 4:
        raise ValueError("Given curve must have more than 4 elements")
    run = x[len(x) - 1] - x[0]
    sample_distance = max(round(len(x) / sampling_size), 1)
    indexes = range(1, len(x) - 1, sample_distance)
    samples = np.array([y[i] for i in indexes], dtype=float)
    total = float(((samples - samples.mean()) ** 2).sum())
    if run == 0 or total == 0:
        #  try inverting x and y, mark not applicable vessels as not tortuous
        return linear_regression_tortuosity(y, x, sampling_size, retry=False) if retry else 1

    slope = (y[len(y) - 1] - y[0]) / run
    y_intercept = y[0] - slope * x[0]
    predicted = np.array([x[i] for i in indexes], dtype=float) * slope + y_intercept
    return 1 - float(((samples - predicted) ** 2).sum()) / total


def distance_measure_tortuosity(x, y):
    """
    Distance measure tortuosity defined in:
    William E Hart, Michael Goldbaum, Brad Côté, Paul Kube, and Mark R Nelson. Measurement and
    classification of retinal vascular tortuosity. International journal of medical informatics,
    53(2):239–252, 1999.

    弧長 ÷ 弦長（Hart 1999 の τ1 に 1 を足した値。本家と同じく −1 はしない）

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the arc-chord tortuosity measure
    """
    if len(x) < 2:
        raise ValueError("Given curve must have at least 2 elements")

    line = _centerline(x, y)
    if line is None:
        return _curve_length(x, y) / _chord_length(x, y)
    return line.length / line.chord


def distance_inflection_count_tortuosity(x, y):
    """
    Calculates the tortuosity by using arc-chord ratio multiplied by the curve inflection count
    plus 1

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the inflection count tortuosity
    """
    return distance_measure_tortuosity(x, y) * (len(_detect_inflection_points(x, y)) + 1)


def fractal_tortuosity(retinal_image: Retina):
    """
    Calculates the fractal dimension of the given image.
    The method used is the Minkowski-Bouligand dimension defined in
    https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension
    :param retinal_image:  a retinal image.
    :return: the fractal dimension of the given image
    """
    return fractal_dimension.fractal_dimension(retinal_image.np_image)


def fractal_tortuosity_curve(x, y):
    image = _curve_to_image(x, y)
    return fractal_dimension.fractal_dimension(image.np_image)


def tortuosity_density(x, y):
    """
    Grisan 2008 の式(19): (n-1)/n × (1/L) × Σ(部分の弧長/部分の弦長 − 1)。n は曲率の符号が一定の部分の数。
    "A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity" by Grisan et al.
    DOI: 10.1109/TMI.2007.904657（学会版は DOI: 10.1109/IEMBS.2003.1279902）

    :param x: the x points of the curve
    :param y: the y points of the curve
    :return: tortuosity density measure
    """
    line = _centerline(x, y)
    if line is None:
        return 0.0
    cuts = _turn_curve_cuts(line)
    n = len(cuts) + 1
    if n == 1:
        return 0.0
    edges = [0] + cuts + [len(line.s) - 1]
    sum_segments = 0.0
    # we process the curve dividing it on its turn curves, the last one included
    for start, end in zip(edges[:-1], edges[1:]):
        chord = float(np.hypot(*(line.xy[end] - line.xy[start])))
        if chord > 0:
            sum_segments += (line.s[end] - line.s[start]) / chord - 1

    return (n - 1) / n * sum_segments / line.length


def squared_curvature_tortuosity(x, y):
    """
    total squared curvature: κ² を弧長で積分した値。Hart 1999 の τ3 は媒介変数 t での ∫κ²dt で、弧長で書くのは
    Grisan 2003 の式(15)。
    See Measurement and classification of retinal vascular tortuosity by Hart et al.
    DOI: 10.1016/S1386-5056(98)00163-4
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the squared curvature tortuosity of the given curve
    """
    line = _centerline(x, y)
    if line is None:
        return 0.0
    return float(_trapezoid(line.kappa ** 2, line.s))


def evaluate_window(window: Window, min_pixels_per_vessel=6, sampling_size=6, r2_threshold=0.80):  # pragma: no cover
    """
    Evaluates a Window object and sets the tortuosity values in the tag parameter.
    :param window: The window object to be evaluated
    :param min_pixels_per_vessel:
    :param sampling_size:
    :param r2_threshold:
    """
    tags = np.empty([window.shape[0], 7])
    # preemptively switch to pytorch.
    window.mode = window.mode_pytorch
    tft = fractal_tortuosity(window)
    for i in range(0, window.shape[0], 1):
        bw_window = window.windows[i, 0, :, :]
        retina = Retina(bw_window, "window{}".format(i) + window.filename)
        retina.threshold_image()
        retina.apply_thinning()
        vessels = detect_vessel_border(retina)
        vessel_count = 0
        t1, t2, t3, t4, td, tfi = 0, 0, 0, 0, 0, 0
        for vessel in vessels:
            if len(vessel[0]) > min_pixels_per_vessel:
                vessel_count += 1
                if linear_regression_tortuosity(vessel[0], vessel[1], sampling_size) > r2_threshold:
                    t1 += 1
                t2 += distance_measure_tortuosity(vessel[0], vessel[1])
                t3 += distance_inflection_count_tortuosity(vessel[0], vessel[1])
                t4 += squared_curvature_tortuosity(vessel[0], vessel[1])
                td += tortuosity_density(vessel[0], vessel[1])
                tfi += fractal_tortuosity_curve(vessel[0], vessel[1])
        if vessel_count > 0:
            t1 = t1/vessel_count
            t2 = t2/vessel_count
            t3 = t3/vessel_count
            t4 = t4/vessel_count
            td = td/vessel_count
            tfi = tfi/vessel_count
        tags[i] = t1, t2, t3, t4, td, tfi, tft
    window.tags = tags
