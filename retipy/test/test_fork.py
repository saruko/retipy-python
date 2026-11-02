"""retipy-python_fork の修正を、真の値または原著の式が分かる入力で確かめるテスト。

本家のテストの多くは出力を csv と比べるだけで、式の誤りを検出できない。ここでは原著の値と比べる。
"""
from unittest import TestCase

import cv2
import numpy as np
from PIL import Image
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

from lib.fractal_dimension import fractal_dimension
from retipy import landmarks as l
from retipy import math as m
from retipy import retina, retina_grayscale, tortuosity, tortuosity_measures as tm
from retipy import vessel_classification as vc
from retipy.retina import Retina, Window, detect_vessel_border

_resources = 'retipy/resources/images/'


def raster(points, shape=(300, 300)):
    image = np.zeros(shape)
    pts = np.round(np.asarray(points)).astype(int)
    for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
        rr, cc = draw_line(a, b, c, d)
        image[rr, cc] = 1
    return image


def true_ratio(points):
    pts = np.asarray(points, float)
    return np.hypot(*np.diff(pts, axis=0).T).sum() / np.hypot(*(pts[-1] - pts[0]))


def longest_vessel(image):
    return max(detect_vessel_border(Retina(skeletonize(image > 0), "t")), key=lambda v: len(v[0]))


class TestDerivatives(TestCase):
    def test_second_difference_of_a_parabola(self):
        self.assertEqual(m.derivative2_centered_h1(4, [i * i for i in range(9)]), 2)

    def test_range_is_checked(self):
        for bad in (0, 2):
            self.assertRaises(ValueError, m.derivative1_centered_h1, bad, [1, 2, 3])
            self.assertRaises(ValueError, m.derivative2_centered_h1, bad, [1, 2, 3])


class TestVesselOrder(TestCase):
    def test_horizontal_vessel_keeps_every_pixel(self):
        vessel = longest_vessel(raster([[150, 20], [150, 280]]))
        self.assertEqual(len(vessel[0]), 261)
        self.assertAlmostEqual(tm.distance_measure_tortuosity(*vessel), 1, places=3)

    def test_points_follow_the_vessel(self):
        t = np.linspace(0, 1, 400)
        vessel = longest_vessel(raster(np.c_[150 + 25 * np.sin(6 * np.pi * t), 20 + 260 * t]))
        steps = np.hypot(np.diff(vessel[0]), np.diff(vessel[1]))
        self.assertLessEqual(steps.max(), 2 ** 0.5 + 1e-9)

    def test_rotating_a_sine_keeps_its_tortuosity(self):
        t = np.linspace(0, 1, 400)
        along = np.c_[20 + 260 * t, 150 + 25 * np.sin(6 * np.pi * t)]
        truth = true_ratio(along)
        for points in (along, along[:, ::-1]):
            self.assertAlmostEqual(tm.distance_measure_tortuosity(*longest_vessel(raster(points))), truth, delta=0.05)

    def test_u_turn_keeps_both_arms(self):
        th = np.linspace(-np.pi / 2, np.pi / 2, 300)
        arc = np.c_[150 + 100 * np.cos(th), 150 + 100 * np.sin(th)]
        vessel = longest_vessel(raster(arc))
        self.assertAlmostEqual(tm.distance_measure_tortuosity(*vessel), true_ratio(arc), delta=0.03)

    def test_junction_splits_into_three(self):
        image = raster([[20, 150], [150, 150]]) + raster([[150, 150], [280, 60]]) + raster([[150, 150], [280, 240]])
        self.assertEqual(len(detect_vessel_border(Retina((image > 0) * 1.0, "t"))), 3)

    def test_small_clump_does_not_drop_the_vessel(self):
        image = raster([[20, 150], [200, 150]])
        image[21, 151] = 1  # 端の近くの小さな塊
        image[20, 151] = 1
        self.assertGreater(len(longest_vessel(image)[0]), 170)
        vessel = detect_vessel_border(Retina(image, "t"))
        self.assertGreater(max(len(v[0]) for v in vessel), 170)

    def test_ring_is_not_returned(self):
        th = np.linspace(0, 2 * np.pi, 400)
        self.assertEqual(detect_vessel_border(Retina(raster(np.c_[150 + 50 * np.sin(th), 150 + 50 * np.cos(th)]), "t")), [])

    def test_input_image_is_not_changed(self):
        image = raster([[20, 150], [150, 150]])
        before = image.copy()
        detect_vessel_border(Retina(image, "t"))
        np.testing.assert_array_equal(image, before)


class TestHart(TestCase):
    def _arc(self, radius=200.0, half=np.deg2rad(45)):
        t = np.linspace(-half, half, 400)
        return list(300 + radius * np.sin(t)), list(300 + radius * np.cos(t))

    def test_total_squared_curvature_of_an_arc(self):
        x, y = self._arc()
        expected = 2 * np.deg2rad(45) * 200.0 / 200.0 ** 2  # ∫κ²ds = L/R²
        self.assertAlmostEqual(tm.squared_curvature_tortuosity(x, y) / expected, 1, delta=0.1)

    def test_s_curve_does_not_cancel(self):
        t = np.linspace(0, 1, 2000)
        x, y = 20 + 260 * t, 150 + 40 * np.sin(2 * np.pi * t)
        dx, dy = np.gradient(x), np.gradient(y)
        kappa = (dx * np.gradient(dy) - dy * np.gradient(dx)) / (dx ** 2 + dy ** 2) ** 1.5
        expected = np.sum(kappa ** 2 * np.hypot(dx, dy))
        self.assertAlmostEqual(tm.squared_curvature_tortuosity(list(x), list(y)) / expected, 1, delta=0.1)


class TestGrisan(TestCase):
    def test_constant_curvature_sign_is_zero(self):
        t = np.linspace(-1, 1, 400)
        self.assertEqual(tm.tortuosity_density(list(300 + 150 * np.sin(t)), list(300 + 150 * np.cos(t))), 0.0)

    def test_equation_19_on_a_sine(self):
        # 3周期の正弦波は曲率の符号が一定の部分が6個（変曲点は両端を除いて5個）
        t = np.linspace(0, 1, 1200)
        x, y = 20 + 400 * t, 200 + 30 * np.sin(6 * np.pi * t)
        cuts = [int(round(k * 1199 / 6)) for k in range(7)]
        steps = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate([[0], np.cumsum(steps)])
        total = sum((s[b] - s[a]) / np.hypot(x[b] - x[a], y[b] - y[a]) - 1 for a, b in zip(cuts[:-1], cuts[1:]))
        expected = 5 / 6 * total / s[-1]
        self.assertAlmostEqual(tm.tortuosity_density(list(x), list(y)) / expected, 1, delta=0.1)

    def test_pixel_stairs_are_not_inflections(self):
        for run in (1, 2, 3):
            x, y = list(range(60)), [i // run for i in range(60)]
            self.assertEqual(tm._detect_inflection_points(x, y), [])
            self.assertEqual(tm.tortuosity_density(x, y), 0.0)

    def test_inflection_of_a_sine(self):
        x = list(range(200))
        self.assertEqual(len(tm._detect_inflection_points(x, [round(40 * np.sin(2 * np.pi * i / 199)) for i in x])), 1)

    def test_straight_line_is_not_flagged_as_tortuous(self):
        result = tortuosity.density(raster([[5, 20], [250, 142]], (256, 256)) * 255, window_size=2, threshold=0.001)
        self.assertEqual(result["data"], [])

    def test_sine_scores_higher_than_a_straight_line(self):
        t = np.linspace(0, 1, 500)
        sine = raster(np.c_[5 + 245 * t, 128 + 40 * np.sin(6 * np.pi * t)], (256, 256)) * 255
        scores = [float(d["notes"]) for d in tortuosity.density(sine, window_size=2)["data"]]
        self.assertGreater(max(scores), 0.001)


class TestDeterminationCoefficient(TestCase):
    def test_never_above_one(self):
        x = list(range(20))
        self.assertLessEqual(tm.linear_regression_tortuosity(x, [i % 2 for i in range(19)] + [19], 6), 1)
        th = np.linspace(0.001, np.pi - 0.001, 40)
        half_circle = tm.linear_regression_tortuosity(list(100 * np.cos(th)[::-1]), list(100 * np.sin(th)), 6)
        self.assertLess(half_circle, 0.5)

    def test_straight_line_is_one(self):
        self.assertAlmostEqual(tm.linear_regression_tortuosity(list(range(30)), [2 * i + 1 for i in range(30)]), 1)

    def test_numpy_input_has_the_same_result(self):
        x = np.arange(30.0)
        self.assertEqual(tm.linear_regression_tortuosity(x, np.ones(30)), 1)


class TestWindowEvaluation(TestCase):
    def _tags(self, curves, size=128):
        image = np.zeros((size, size))
        for curve in curves:
            image += raster(curve, (size, size))
        window = Window(Retina((image > 0) * 1.0, "w"), size, min_pixels=1)
        tm.evaluate_window(window, 6, 6, 0.8)
        return window

    def test_all_measures_are_averaged_over_vessels(self):
        t = np.linspace(0, 1, 200)
        curve = np.c_[5 + 110 * t ** 2, 10 + 40 * t]
        one = self._tags([curve]).tags[0]
        two = self._tags([curve, curve + [0, 60]]).tags[0]
        self.assertGreater(one[3], 0)
        self.assertGreater(one[5], 0)
        for column in (1, 3, 4, 5):
            self.assertAlmostEqual(one[column], two[column], delta=abs(one[column]) * 0.02 + 1e-6)

    def test_tag_layer_is_added_for_seven_tags(self):
        t = np.linspace(0, 1, 200)
        window = self._tags([np.c_[5 + 110 * t, 10 + 40 * t]])
        before = window.windows.copy()
        window.set_tag_layer()
        self.assertEqual(window.windows.shape[1], 2)
        np.testing.assert_array_equal(window.windows[:, 0], before[:, 0])
        window.set_tag_layer()
        self.assertEqual(window.windows.shape[1], 2)


class TestFractalDimension(TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:256, :256]
        self.disc = ((yy - 128) ** 2 + (xx - 128) ** 2) < 90 ** 2

    def test_filled_disc_is_close_to_two(self):
        self.assertGreater(fractal_dimension(self.disc), 1.7)

    def test_line_is_close_to_one(self):
        self.assertAlmostEqual(fractal_dimension(raster([[5, 5], [250, 180]], (256, 256))), 1, delta=0.15)

    def test_value_range_does_not_matter(self):
        expected = fractal_dimension(self.disc)
        for image in (self.disc.astype(float), self.disc.astype(np.uint8), self.disc.astype(np.uint8) * 255):
            self.assertEqual(fractal_dimension(image), expected)

    def test_blank_image_raises(self):
        self.assertRaises(ValueError, fractal_dimension, np.zeros((64, 64)))

    def test_curve_input_is_not_changed(self):
        x, y = [10, 11, 12, 13, 14, 15], [50, 51, 52, 53, 54, 55]
        tm.fractal_tortuosity_curve(x, y)
        self.assertEqual((x, y), ([10, 11, 12, 13, 14, 15], [50, 51, 52, 53, 54, 55]))


class TestLandmarks(TestCase):
    def _thick(self, points, shape=(240, 240)):
        return cv2.dilate((raster(points, shape) * 255).astype(np.uint8), np.ones((3, 3), np.uint8))

    def test_vessel_width_is_the_true_width(self):
        for width in (1, 3, 5, 7, 9):
            image = np.zeros((60, 60), np.uint8)
            image[:, 30 - width // 2:30 + width // 2 + 1] = 255
            result = l.vessel_width(image, [[30, 30]])[0]
            self.assertEqual(result[1] + result[2], width)

    def test_two_pixel_vessel_on_either_side(self):
        for columns in (slice(30, 32), slice(29, 31)):
            image = np.zeros((60, 60), np.uint8)
            image[:, columns] = 255
            result = l.vessel_width(image, [[30, 30]])[0]
            self.assertEqual(result[1] + result[2], 2)

    def test_crossing_box_does_not_depend_on_the_border(self):
        crossing = np.maximum(self._thick([[40, 40], [200, 200]]), self._thick([[40, 200], [200, 40]]))
        boxes = [l.classification(crossing.copy(), border)[1] for border in (0, 2, 20)]
        self.assertEqual(boxes[0], boxes[1])
        self.assertEqual(boxes[0], boxes[2])
        box = boxes[0][0]
        self.assertEqual((box[2] - box[0], box[3] - box[1]), (6, 6))
        self.assertLessEqual(abs((box[0] + box[2]) / 2 - 120), 4)

    def test_vessels_at_the_image_edge_with_no_border(self):
        manual = np.array(Image.open(_resources + "manual.png").convert("L"))
        l.classification(manual, 0)


class TestSegmentation(TestCase):
    def setUp(self):
        self.rgb = np.array(Image.open(_resources + "original.tif").convert("RGB"))

    def test_outside_of_the_field_of_view_gets_the_mean(self):
        captured = {}

        class Probe(retina_grayscale.Retina_grayscale):
            def median_filter(self, structure):
                captured["image"] = self.np_image.copy()
                captured["mean"] = self.mean_value
                super().median_filter(structure)

        probe = Probe(self.rgb.copy(), None)
        probe.shadow_correction()
        outside = captured["image"][probe.mask == 0]
        self.assertTrue((outside == int(captured["mean"])).all())

    def test_homogenized_background_is_128(self):
        image = retina_grayscale.Retina_grayscale(self.rgb.copy(), None)
        image.shadow_correction()
        image.homogenize()
        values = image.np_image[image.mask == 1].astype(np.int64)
        self.assertEqual(np.bincount(values).argmax(), 128)

    def test_post_processing_keeps_the_size(self):
        image = retina_grayscale.Retina_grayscale(self.rgb.copy(), None)
        mask = np.zeros(image.shape)
        mask[50:60, 20:200] = 255
        self.assertEqual(image.post_processing(mask).shape, image.shape)


class TestClassification(TestCase):
    def setUp(self):
        self.bgr = cv2.imread(_resources + "original.tif", 1)
        self.av = cv2.imread(_resources + "av.png", 1)
        self.manual = np.array(Image.open(_resources + "manual.png").convert("L"))

    def test_bgr_input_finds_arteries_and_veins(self):
        post = vc.classification(self.bgr, self.manual.copy())
        _, segments, _, _ = vc._loading_model(self.bgr, self.manual.copy(), self.av, 38)
        self.assertGreater(vc._accuracy(post, segments, self.av)[0], 0.8)
        self.assertGreater((post == [0, 0, 255]).all(-1).sum(), 0)
        self.assertGreater((post == [255, 0, 0]).all(-1).sum(), 0)

    def test_connected_components_are_eight_connected(self):
        line = raster([[10, 10], [60, 60]], (70, 70)).astype(np.uint8)
        self.assertEqual(cv2.connectedComponentsWithStats(line, connectivity=8, ltype=cv2.CV_32S)[0] - 1, 1)


class TestImageOutput(TestCase):
    def test_float_image_is_not_black(self):
        import base64
        import io
        rgb = (np.random.RandomState(0).rand(16, 16, 3) * 255).astype(np.uint8)
        data = base64.b64decode(Retina(rgb, "x").original_base64)
        self.assertGreater(np.array(Image.open(io.BytesIO(data))).max(), 100)

    def test_compare_binary_images(self):
        a, b = Retina(np.ones((4, 4)), "a"), Retina(np.zeros((4, 4)), "b")
        a.np_image, b.np_image = a.np_image > 0, b.np_image > 0
        self.assertEqual(a.compare_with(b).np_image.sum(), 16)

    def test_save_window_uses_the_folder(self):
        import os
        import tempfile
        folder = tempfile.mkdtemp()
        window = Window(Retina(np.ones((16, 16), np.uint8), "w.png"), 8, min_pixels=0)
        window.windows = window.windows.astype(np.uint8)
        window.save_window(0, folder)
        self.assertTrue(os.path.isfile(os.path.join(folder, "out_w0_w.png")))
