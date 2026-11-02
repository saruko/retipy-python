# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017-2018  Alejandro Valdes
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

"""retina module to handle basic image processing on retinal images

Modified 2026-11-02 for retipy-python_fork. See README_fork.md for the list of changes.
"""

import base64
import math
import numpy as np
import warnings
from copy import copy
from io import BytesIO
from lib import thinning
from matplotlib import pyplot as plt
from os import path
from PIL import Image
from scipy import ndimage
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize


class Retina(object):
    """
    Retina class that internally contains a matrix with the image data for a retinal image, it
    constructor expects a path to the image

    :param image: a numpy array with the image data
    :param image_path: path to an image to be open
    """
    @staticmethod
    def _open_image(img_path):
        return io.imread(img_path)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """0〜1 の小数や二値の画像は、そのまま uint8 にするとほぼ真っ黒になるので 255 倍する"""
        if image.dtype == bool or (image.dtype.kind == 'f' and image.max() <= 1):
            image = image * 255.0
        return np.clip(image, 0, 255).astype('uint8')

    @staticmethod
    def get_base64_image(image: np.ndarray, is_luminance: bool = True):
        if is_luminance:
            temp_image = Image.fromarray(Retina._to_uint8(image))
        else:
            temp_image = Image.fromarray(image)
        buffer = BytesIO()
        temp_image.save(buffer, format="png")
        return str(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    def __init__(self, image: np.ndarray, image_path: str):
        if image is None:
            self.np_image = self._open_image(image_path)
            _, file = path.split(image_path)
            self._file_name = file
        else:
            self.np_image = image
            self._file_name = image_path

        self.segmented = False
        self.old_image = None
        if self.np_image.ndim == 3:
            self.np_image = color.rgb2gray(self.np_image[:, :, :3])
        self.original_base64 = self.get_base64_image(self.np_image)
        self.depth = 1
        self.shape = self.np_image.shape

##################################################################################################
# Image Processing functions

    def threshold_image(self):
        """Applies a thresholding algorithm to the contained image."""
        threshold = filters.threshold_mean(self.np_image)
        self.np_image = self.np_image > threshold
        self.depth = 1

    def detect_edges_canny(self, min_val=0, max_val=1):
        """
        Applies canny edge detection to the contained image. Fine tuning of the algorithm can be
        done using min_val and max_val
        """
        self._copy()
        self.np_image = feature.canny(self.np_image, low_threshold=min_val, high_threshold=max_val)

    def apply_thinning(self):
        """Applies a thinning algorithm on the stored image"""
        self._copy()
        self.np_image = thinning.thinning_zhang_suen(self.np_image)

    def erode(self, times):
        """
        Erodes the stored image
        :param times: number of times that the image will be eroded
        """
        self._copy()
        self.np_image = ndimage.binary_erosion(self.np_image, iterations=times)

    def dilate(self, times):
        """
        dilates the stored image
        :param times: number of times that the image will be dilated
        """
        self._copy()
        self.np_image = ndimage.binary_dilation(self.np_image, iterations=times)

    def reshape_square(self):
        """
        This function will normalise the image size, making a square with it and rounding the pixels:
        If the given image is 571 560, the new size will be 572 572, with zeroes in all new pixels.
        """
        max_value = self.shape[0] if self.shape[0] > self.shape[1] else self.shape[1]
        max_value = max_value + (max_value % 2)
        self.np_image = np.pad(
            self.np_image,
            ((0, max_value - self.shape[0]), (0, max_value - self.shape[1])),
            'constant',
            constant_values=(0, 0))
        self.shape = self.np_image.shape

    def reshape_by_window(self, window: int, is_percentage: bool=False) -> int:
        """
        Reshapes the internal image to be able to be divided by the given window size
        :param window: an integer with the window size. Considered as a square
        :param is_percentage: sets if the given window is a percentage of the image or a pixel value

        :return: the dimension of the image on which was resized.
        """
        dimension = window
        if is_percentage:
            # get the smallest dimension
            selected_dimension = self.shape[0] if self.shape[0] < self.shape[1] else self.shape[1]
            dimension = int(math.floor(selected_dimension/window))
            # make it even
            dimension += dimension % 2
            x_pixels = (math.ceil(self.shape[0]/dimension)*dimension) - self.shape[0]
            y_pixels = (math.ceil(self.shape[1]/dimension)*dimension) - self.shape[1]
        else:
            x_pixels = (math.ceil(self.shape[0] / window) * window) - self.shape[0]
            y_pixels = (math.ceil(self.shape[1] / window) * window) - self.shape[1]
        self.np_image = np.pad(
            self.np_image, ((0, x_pixels), (0, y_pixels)), 'constant', constant_values=(0, 0))
        self.shape = self.np_image.shape

        return dimension

    def get_window_sizes(self):
        """
        Returns an array with the possible window size that this image can be divided by without leaving empty space.
        584x584 would return [292,146,73]
        This is only available for square images (you can use reshape_square() before calling this method)
        :return: a list of possible window sizes.
        """
        sizes = []
        if self.shape[0] == self.shape[1]:
            current_value = self.shape[0]
            while current_value % 2 == 0:
                current_value = current_value // 2
                sizes.append(current_value)
        return sizes

    def skeletonization(self):
        """Applies a skeletonization algorithm to the contained image."""
        self.np_image = skeletonize(self.np_image)

    def bin_to_bgr(self):
        """Transform the image to a ndarray with depth:3"""
        h, w = self.np_image.shape
        image_bgr = np.zeros((h, w, 3))
        image_bgr[:, :, 0] = self.np_image
        image_bgr[:, :, 1] = self.np_image
        image_bgr[:, :, 2] = self.np_image
        self.np_image = image_bgr

    def get_uint_image(self):
        """
        Returns the np_image converted to uint8 and multiplied by 255 to simulate grayscale
        :return: a ndarray image
        """
        image = self.np_image.astype(np.uint8) * 255
        return image

    def reshape_for_landmarks(self, size: int):
        """
        Reshapes the internal image to fix erros with retinal images without borders
        :param size: an integer with the border size.
        """
        self.np_image = np.pad(self.np_image, pad_width=size, mode='constant', constant_values=0)
        self.shape = self.np_image.shape


##################################################################################################
# I/O functions

    def _copy(self):
        self.old_image = copy(self.np_image)

    def undo(self):
        """
        Reverts the latest modification to the internal image, useful if you are testing different
         values
        """
        self.np_image = self.old_image

    @property
    def filename(self):
        """Returns the filename of the retina image."""
        return self._file_name

    def _output_filename(self):
        return "out_" + self.filename

    def save_image(self, output_folder):
        """Saves the image in the given output folder, the name will be out_<original_image_name>"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(path.join(output_folder, self._output_filename()), self._to_uint8(self.np_image))

    def view(self):  # pragma: no cover
        """show a window with the internal image"""
        io.imshow(self.np_image)
        plt.show()

    def compare_with(self, retinal_image):
        """
        Returns the difference between the given image and the stored one.

        :param retinal_image: the image to compare with
        :return:  a new Retina object with the differences
        """
        difference = self.np_image.astype(float) - retinal_image.np_image.astype(float)
        return Retina(difference, "diff" + self.filename)


class Window(Retina):
    """
    a ROI (Region of Interest) that extends the Retina class
    TODO: Add support for more than depth=1 images (only if needed)
    """
    def __init__(self, image: Retina, dimension, method="separated", min_pixels=10):
        super(Window, self).__init__(
            image.np_image,
            image.filename)
        self.windows, self.w_pos = Window.create_windows(image, dimension, method, min_pixels)
        if len(self.windows) == 0:
            raise ValueError("No windows were created for the given retinal image")
        self.shape = self.windows.shape
        self._mode = self.mode_pytorch
        self._tags = None
        self._has_tag_layer = False

    @property
    def mode_pytorch(self):
        return "PYT"

    @property
    def mode_tensorflow(self):
        return "TF"

    @property
    def tags(self) -> np.ndarray:
        return self._tags

    @tags.setter
    def tags(self, value: np.ndarray):
        if value.shape[0] != self.shape[0]:
            raise ValueError(
                "Wrong set of tags, expected {} got {}".format(self.shape[0], value.shape[0]))
        self._tags = value

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode):
        """
        Changes the internal window ordering depending on the given mode.
        Tensorflow style is [batch, width, height, depth]
        Pytorch style is [batch, depth, width, height]
        :param mode: new mode to change, can be self.tensorflow or self.pytorch
        """
        if mode == self.mode_pytorch and self._mode == self.mode_tensorflow:
            twin = np.swapaxes(self.windows, 2, 3)
            self.windows = np.swapaxes(twin, 1, 2)
            self.shape = self.windows.shape
            self._mode = self.mode_pytorch
        elif mode == self.mode_tensorflow and self._mode == self.mode_pytorch:
            twin = np.swapaxes(self.windows, 1, 2)
            self.windows = np.swapaxes(twin, 2, 3)
            self.shape = self.windows.shape
            self._mode = self.mode_tensorflow

    def _window_filename(self, window_id):
        return "out_w" + str(window_id) + "_" + self.filename

    def save_window(self, window_id, output_folder):
        """
        Saves the specified window in the given output folder, the name will be out_<original_image_name>
        :param window_id: the window_id to save
        :param output_folder: destination folder
        """
        if window_id >= self.windows.shape[0]:
            raise ValueError(
                "Window value '{}' is more than allowed ({})".format(
                    window_id, self.windows.shape[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(
                path.join(output_folder, self._window_filename(window_id)), self._to_uint8(self.windows[window_id, 0]))

    class _WindowIterator:
        def __init__(self, window: np.ndarray):
            self.__window = window
            self.__iterator = 0

        def __len__(self):
            return self.__window.shape[0]

        def __next__(self):
            self.__iterator += 1
            if self.__iterator <= self.__len__():
                return self.__window[self.__iterator - 1]
            else:
                raise StopIteration

    def __iter__(self):
        return self._WindowIterator(self.windows)

    @staticmethod
    def _create_tag_image(dx: int, dy: int, tags: list) -> np.ndarray:
        image = np.zeros((dx, dy))
        boxes = len(tags) % 2 + len(tags)
        box_count = (boxes // 2)
        tag_x = dx // box_count
        tag_y = dy // 2
        # タグが奇数個のときは最後の箱が余るので、0 のままにする
        for i in range(0, len(tags)):
            cx = (i % box_count) * tag_x
            cy = (i // box_count) * tag_y
            image[cx:cx+tag_x, cy:cy+tag_y] = tags[i]
        return image

    def view_window(self, w_id, layer):  # pragma: no cover
        io.imshow(self.windows[w_id, layer])
        plt.show()

    def set_tag_layer(self):
        """
        this method adds the tags values as a new depth window.
        The image will be splitted by the number of tags, resulting in a rectangle per
        tag which will contain the tag value repeated on it.
        """
        if self.tags is None:
            raise ValueError("tags is not set")
        if self.mode != self.mode_pytorch:
            self.mode = self.mode_pytorch

        layer = np.stack([
            self._create_tag_image(self.windows.shape[2], self.windows.shape[3], tags)
            for tags in self.tags])
        if self._has_tag_layer:
            self.windows[:, -1] = layer
        else:
            self.windows = np.concatenate([self.windows, layer[:, np.newaxis]], axis=1)
            self.shape = self.windows.shape
            self.depth += 1
            self._has_tag_layer = True

    @staticmethod
    def create_windows(
            image: Retina, dimension, method="separated", min_pixels=10) -> tuple:
        """
        Creates multiple square windows of the given dimension for the current retinal image.
        Empty windows (i.e. only background) will be ignored

        Separated method will create windows of the given dimension size, that does not share any
        pixel, combined will make windows advancing half of the dimension, sharing some pixels
        between adjacent windows.
        :param image: an instance of Retina, to be divided in windows
        :param dimension:  window size (square of [dimension, dimension] size)
        :param method: method of separation (separated or combined)
        :param min_pixels: ignore windows with less than min_pixels with value.
                           Set to zero to add all windows
        :return: a tuple with its first element as a numpy array with the structure
                 [window, depth, height, width] and its second element as [window, 2, 2]
                 with the window position
        """
        if image.shape[0] % dimension != 0 or image.shape[1] % dimension != 0:
            raise ValueError(
                "image shape is not the same or the dimension value does not divide the image "
                "completely: sx:{} sy:{} dim:{}".format(image.shape[0], image.shape[1], dimension))

        #                      window_count
        windows = []
        windows_position = []
        window_id = 0
        img_dimension = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
        if method == "separated":
            windows = np.empty(
                [(img_dimension // dimension) ** 2, image.depth, dimension, dimension])
            windows_position = np.empty([(img_dimension // dimension) ** 2, 2, 2], dtype=np.int64)
            for x in range(0, image.shape[0], dimension):
                for y in range(0, image.shape[1], dimension):
                    cw = windows_position[window_id]
                    cw[0, 0] = x
                    cw[1, 0] = x + dimension
                    cw[0, 1] = y
                    cw[1, 1] = y + dimension
                    t_window = image.np_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        window_id += 1
        elif method == "combined":
            new_dimension = dimension // 2
            windows = np.empty(
                [(img_dimension // new_dimension) ** 2, image.depth, dimension, dimension])
            windows_position = np.empty([(img_dimension // new_dimension) ** 2, 2, 2], dtype=np.int64)
            if dimension % 2 != 0 or image.shape[0] % new_dimension != 0:
                raise ValueError(
                    "Dimension value '{}' is not valid, choose a value that its half value can split the image evenly"
                    .format(dimension))
            for x in range(0, image.shape[0] - new_dimension, new_dimension):
                for y in range(0, image.shape[1] - new_dimension, new_dimension):
                    cw = windows_position[window_id]
                    cw[0, 0] = x
                    cw[1, 0] = x + dimension
                    cw[0, 1] = y
                    cw[1, 1] = y + dimension
                    t_window = image.np_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        window_id += 1
        if window_id <= windows.shape[0]:
            if window_id == 0:
                windows = []
                windows_position = []
            else:
                windows = np.resize(
                    windows, [window_id, windows.shape[1], windows.shape[2], windows.shape[3]])
                windows_position = np.resize(windows_position, [window_id, 2, 2])

        #  print('created ' + str(window_id) + " windows")
        return windows, windows_position


_NEIGHBOUR_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _intersection_number(skeleton: np.ndarray, x: int, y: int) -> int:
    """Calvo 2011 の式(3)。周囲8画素を一周して、血管が何本出ているかを数える。分岐と交叉で 3 以上になる"""
    values = [skeleton[x + dx, y + dy]
              if 0 <= x + dx < skeleton.shape[0] and 0 <= y + dy < skeleton.shape[1] else False
              for dx, dy in _NEIGHBOUR_RING]
    return sum(values[i] != values[(i + 1) % 8] for i in range(8)) // 2


def _remove_corner_pixels(skeleton: np.ndarray) -> np.ndarray:
    """階段の角の余分な画素を除く。除いても周囲のつながりが変わらない画素だけを除くので、血管は切れない"""
    thin = skeleton.copy()
    connected = np.ones((3, 3))
    removed = True
    while removed:
        removed = False
        for x, y in zip(*np.nonzero(thin)):
            around = thin[max(x - 1, 0):x + 2, max(y - 1, 0):y + 2].copy()
            if around.sum() < 4 or _intersection_number(thin, x, y) != 2:
                continue
            around[min(x, 1), min(y, 1)] = False
            if ndimage.label(around, structure=connected)[1] == 1:
                thin[x, y] = False
                removed = True
    return thin


def _order_path(pixels: list):
    """
    8近傍でつながる画素のうち、最も離れた2点を結ぶ経路を順に返す。端点のない輪は None を返す。
    細線化で残る小さな塊があっても、血管全体を捨てずに本幹をたどる。
    """
    members = set(pixels)
    if len(members) == 1:
        return list(members)
    neighbours_of = {
        p: [(p[0] + dx, p[1] + dy) for dx, dy in _NEIGHBOUR_RING if (p[0] + dx, p[1] + dy) in members]
        for p in members}
    ends = sorted(p for p, n in neighbours_of.items() if len(n) == 1)
    if not ends:
        return None

    def farthest_from(start):
        parents = {start: None}
        queue = [start]
        for current in queue:
            for following in neighbours_of[current]:
                if following not in parents:
                    parents[following] = current
                    queue.append(following)
        return queue[-1], parents

    first, _ = farthest_from(ends[0])
    current, parents = farthest_from(first)
    ordered = []
    while current is not None:
        ordered.append(current)
        current = parents[current]
    return ordered


def detect_vessel_border(image: Retina, ignored_pixels=1):
    """
    Extracts the vessel border of the given image, this method will try to extract all vessel
    borders that does not overlap.

    骨格を分岐点で区切り、各区間の画素を端点から血管に沿った順に返す。端点のない輪は返さない。
    入力の画像は書き換えない。

    Returns a list of lists with the points of each vessel.

    :param image: the retinal image to extract its vessels
    :param ignored_pixels: how many pixels will be ignored from borders. A vessel found only inside
                           this border is not returned.
    """
    skeleton = _remove_corner_pixels(np.asarray(image.np_image) > 0)
    segments = skeleton.copy()
    for x, y in zip(*np.nonzero(skeleton)):
        if _intersection_number(skeleton, x, y) > 2:
            segments[max(x - 1, 0):x + 2, max(y - 1, 0):y + 2] = False

    labels, _ = ndimage.label(segments, structure=np.ones((3, 3)))
    rows, cols = np.nonzero(labels)
    pixels_of = {}
    for x, y in zip(rows.tolist(), cols.tolist()):
        pixels_of.setdefault(labels[x, y], []).append((x, y))

    vessels = []
    for label in sorted(pixels_of):
        pixels = pixels_of[label]
        if not any(ignored_pixels <= x < labels.shape[0] - ignored_pixels
                   and ignored_pixels <= y < labels.shape[1] - ignored_pixels for x, y in pixels):
            continue
        ordered = _order_path(pixels)
        if ordered is not None:
            vessels.append([[p[0] for p in ordered], [p[1] for p in ordered]])
    return vessels
