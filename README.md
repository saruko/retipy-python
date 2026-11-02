# retipy-python (Modified Fork)

This is a modified fork of [alevalv/retipy-python](https://github.com/alevalv/retipy-python) (commit `e6f0ed2`, 2019-05-18), corrected to align with the mathematical and algorithmic definitions in the original papers.

このリポジトリは [alevalv/retipy-python](https://github.com/alevalv/retipy-python)（コミット `e6f0ed2`、2019-05-18）を、原著論文および数学的定義に合致するよう修正したフォーク版です。

- [English: Summary of Modifications & References](#english)
- [日本語: 改正点と根拠（引用文献）](#japanese)
- [Original README](#original-readme)

---

<a id="english"></a>
## English: Summary of Modifications & References

### Revision Policy
1. **Fix errors verifiable against original papers or mathematical definitions.** Every modification is backed by explicit equations or definitions in the cited literature.
2. **Do not change code without clear evidence.** Components whose source literature could not be identified or where changes degrade performance are kept as-is, with reasons documented below.
3. **Preserve feature definitions assumed by pretrained weights.** Modifying feature scaling or sign conventions used during training invalidates the bundled artery/vein classification weights.
4. **Verify against ground-truth analytical inputs.** Upstream unit tests largely compare outputs against pre-recorded CSV snapshots and cannot detect formula errors. We added `retipy/test/test_fork.py` to test against analytical curves and mathematical ground truths.

### References
| Short cite | Full citation | Used for |
|:---|:---|:---|
| **Hart 1999** | Hart WE, Goldbaum M, Côté B, Kube P, Nelson MR. Measurement and classification of retinal vascular tortuosity. *Int J Med Inform*. 1999;53(2-3):239–252. doi:10.1016/S1386-5056(98)00163-4 | Eq. (2), Table 1 & definition of total squared curvature ($tsc$) on p. 244, centerline smoothing on p. 245 |
| **Grisan 2003** | Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic evaluation of retinal vessel tortuosity. *Proc 25th Annu Int Conf IEEE EMBS*. 2003;1:866–869. doi:10.1109/IEMBS.2003.1279902 | Eqs. (13)–(15) (arc-length parameterization), curvature hysteresis threshold |
| **Grisan 2008** | Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic grading of retinal vessel tortuosity. *IEEE Trans Med Imaging*. 2008;27(3):310–319. doi:10.1109/TMI.2007.904657 | Eqs. (17)–(19) and p. 314 (inflection point splitting and tortuosity density $TD$) |
| **Marín 2011** | Marín D, Aquino A, Gegúndez-Arias ME, Bravo JM. A new supervised method for blood vessel segmentation in retinal images by using gray-level and moment invariants-based features. *IEEE Trans Med Imaging*. 2011;30(1):146–158. doi:10.1109/TMI.2010.2064333 | Eqs. (1)–(3), p. 149 (FOV border padding and background homogenization) |
| **Calvo 2011** | Calvo D, Ortega M, Penedo MG, Rouco J. Automatic detection and characterisation of retinal vessel tree bifurcations and crossovers in eye fundus images. *Comput Methods Programs Biomed*. 2011;103(1):28–38. doi:10.1016/j.cmpb.2010.06.002 | Eq. (3) (crossing number $CN(p)$ for skeleton bifurcation splitting) |
| **Minkowski–Bouligand** | Wikipedia contributors. Minkowski–Bouligand dimension (Box-counting dimension). Cited in upstream `fractal_dimension.py`. | Definition of box-counting dimension ($N(\varepsilon)$ counts all boxes covering the set) |

*Note:* Upstream explicitly cites only Hart 1999, Grisan 2003, and Wikipedia. Marín 2011 and Calvo 2011 are identified based on algorithmic correspondence in preprocessing and skeleton crossing numbers.

### List of Modifications

#### 1. Tortuosity (`tortuosity_measures.py`, `retina.py`, `tortuosity.py`, `math.py`)
| Target | Upstream behavior | Modified behavior | Rationale & Citation | Verification (Upstream → This fork) |
|:---|:---|:---|:---|:---|
| **Point ordering** (`detect_vessel_border`) | Sorts skeleton pixels by $x$-coordinate, keeps at most 1 pixel per row, and treats the entire connected tree as a single curve. | Splits the skeleton at junctions ($CN(p) \ge 3$, Calvo 2011 Eq. 3) and traces each simple segment sequentially from endpoint to endpoint. Closed loops are excluded. | **Hart 1999, p. 245:** Arc length is the sum of Euclidean distances between adjacent points ordered sequentially along the vessel curve. Sorting by $x$ collapses horizontal vessels and scrambles branching trees. | Horizontal 261-px line: 1 pt → 261 pts.<br>Sine wave (true ratio 1.582): vertical 1.625, horizontal 1.069 → 1.573 in both orientations.<br>Semicircle (true 1.571): 1.576 / 1.143 → 1.593 in both. |
| **Curve representation** | Differentiates raw discrete pixel coordinates directly. | Fits a parametric smoothing spline (`Centerline` via `scipy.interpolate.splprep`, $s = 0.35 N$) before computing arc length and curvature. | **Hart 1999, p. 245; Grisan 2003; Grisan 2008:** Differentiating raw pixel staircases overwhelms true vessel curvature with discretization noise. | Verified across all downstream curvature metrics below. |
| **Total squared curvature** (`squared_curvature_tortuosity`) | Computes $\left|\int \kappa \, dt\right|$ (unsigned integral of signed curvature without squaring, where opposite bends cancel out). | Computes $\int \kappa(s)^2 \, ds$ (integral of squared curvature with respect to arc length $s$). | **Hart 1999, p. 244 Table 1 ($tsc$); Grisan 2003, Eq. (15):** Total squared curvature integrates $\kappa^2$. | Symmetric S-curve (true 0.1461): 0.0000 → 0.1434.<br>Circular arc (true 0.0149): 0.3412 → 0.0150. |
| **2nd-order finite difference** (`derivative2_centered_h1`) | Divides $y_{i+1} - 2y_i + y_{i-1}$ by 4. | Does not divide by 4 (uses step size $h=1$: $(y_{i+1} - 2y_i + y_{i-1}) / 1^2$). | Standard centered second difference for unit spacing $h=1$. | Second derivative of $y = x^2$ (true value 2): 0.5 → 2.0. |
| **Inflection points** (`_detect_inflection_points`) | Counts local extrema where raw $y$-coordinates change monotonicity ($\Delta y$ sign flips). | Detects sign changes in spline curvature $\kappa(s)$ exceeding a hysteresis threshold ($0.01\text{ px}^{-1}$) and places the inflection point at the midpoint of the zero-curvature band. | **Grisan 2008, Eq. (17) & p. 314; Grisan 2003:** Inflection points are where curvature $\kappa$ changes sign, not where $y(t)$ has a horizontal tangent. Hysteresis suppresses staircase noise. | Straight line of slope 1/2 (true 0): 58 → 0.<br>Rasterized sine wave with 1 inflection (true 1): 56 → 1. |
| **Tortuosity density** (`tortuosity_density`) | Computes $\frac{n-1}{n} + \frac{1}{L}\sum \dots$, where $n$ is the number of inflection points, omitting the final subsegment and segments immediately preceding each inflection point. | Computes $\frac{n-1}{n} \cdot \frac{1}{L} \sum_{i=1}^{n} \left(\frac{L_{cs_i}}{L_{xs_i}} - 1\right)$, where $n$ is the number of constant-sign curvature subsegments, summing over all $n$ subsegments ($0$ when $n \le 1$). | **Grisan 2008, Eqs. (18)–(19):** The factor $\frac{n-1}{n}$ multiplies the sum (so a single-arch curve with $n=1$ has $TD=0$), and $n$ is the number of subsegments partitioned by $n-1$ inflection points. | Straight line of slope 1/2: 0.9828 → 0.0.<br>Sine wave (analytical Eq. 19 value 0.00351): 0.00346. |
| **Window filtering** (`tortuosity.density`, `fractal`) | Hardcodes default thresholds `min_val=0.97` and `0.94`, discarding windows below them. | Defaults to `min_val=None` (returns all windows) and increases output precision. | The threshold `0.97` was an artifact of the broken additive formula $\frac{n-1}{n} + \dots$ and does not exist in Grisan 2008. | Upstream flagged straight lines as tortuous (0.98) and rejected sine waves; now outputs 0.00000 for straight lines and up to 0.0049 for sine waves. |
| **Coefficient of determination $R^2$** (`linear_regression_tortuosity`) | Computes $\text{SSR} / \text{SST} = \sum (\hat{y} - \bar{y})^2 / \sum (y - \bar{y})^2$ on the chord connecting the two endpoints. | Computes $1 - \text{SSE} / \text{SST} = 1 - \sum (y - \hat{y})^2 / \sum (y - \bar{y})^2$. | **Definition of $R^2$:** The identity $\text{SST} = \text{SSR} + \text{SSE}$ holds only for OLS regression lines. Because the chord between endpoints is not an OLS fit, $\text{SSR}/\text{SST}$ can exceed 1 for highly curved vessels. | Zigzag curve: 361 → −367.<br>Semicircle: 3.14 → −3.14.<br>Straight line: remains 1.0. |
| **Window averaging** (`evaluate_window`) | Divides metrics by vessel count, except `t4` (`linear_regression_tortuosity`) and `tfi` (`distance_inflection_count_tortuosity`), which are left as raw sums. | Divides all per-vessel metrics (`t4` and `tfi` included) by `vessel_count`. | Consistency with the other per-window mean tortuosity columns. | Duplicating an identical vessel in a window doubled `t4` (0.228 → 0.456) and `tfi` (0.935 → 1.870); now remains constant (0.0053 and 0.9305). |
| **Stub function** (`smooth_tortuosity_cubic`) | Unfinished stub returning `y[0]`. | Removed. | Dead incomplete code. | — |

#### 2. Fractal Dimension (`retipy/lib/fractal_dimension.py`)
| Target | Upstream behavior | Modified behavior | Rationale & Citation | Verification |
|:---|:---|:---|:---|:---|
| **Box counting** (`fractal_dimension`) | Counts only partially filled boxes (`(S > 0) & (S < k*k)`), ignoring completely filled interior boxes, and thresholds using the raw pixel intensity sum. | Binarizes the foreground (`b_image > 0`) and counts every box containing at least one foreground pixel (`S > 0`). | **Minkowski–Bouligand dimension:** Box-counting dimension requires counting the total number of boxes $N(\varepsilon)$ needed to cover the set, not just boundary boxes. | Filled disk: 1.148 → 1.750.<br>0/255 binary image: `NaN` → 1.750.<br>1-px line: remains 1.082. |

#### 3. Landmarks: Bifurcations and Crossings (`landmarks.py`)
| Target | Upstream behavior | Modified behavior | Rationale & Citation | Verification |
|:---|:---|:---|:---|:---|
| **Crossing bounding box** (`find_landmarks`) | Uses `+ border` instead of `- border` for the bottom-right coordinates of crossing boxes (`x + s + border`), unlike bifurcation boxes (`x + s - border`). | Uses `- border` consistently so box coordinates correctly map back to unpadded image coordinates. | Coordinate offset bug when stripping padding of width `border`. | Crossing box width for `border = 0, 2, 20`: 6, 10, 46 px → 6 px in all cases. |
| **Vessel width** (`vessel_width`) | Skips the first pixel on the opposite ray and excludes the center skeleton pixel, undercounting width by 1 px whenever diameter $> 1$. | Includes the center pixel so that opposite rays sum to the true vessel width. | Geometric definition of width across a skeleton point. | True widths 1, 3, 5, 7, 9 px: 1, 2, 4, 6, 8 → 1, 3, 5, 7, 9. |
| **Image boundaries** (`pixel_value`, `skeleton_Connectivity`) | Negative indices wrap around to the opposite side of the NumPy array; empty ROI slices crash OpenCV (`cv2.connectedComponentsWithStats`). | Treats out-of-bounds coordinates as background (`0`) and guards against empty ROIs. | Prevents wrap-around artifacts and OpenCV 4.7+ crashes when `border=0`. | No longer crashes on border vessels with `border=0`. |
| **Connected components argument** | Calls `connectedComponentsWithStats(img, 4, cv2.CV_32S)`, passing `4` as the positional `labels` output array rather than `connectivity`, so OpenCV silently defaults to 8-connectivity. | Explicitly passes `connectivity=8` to match actual runtime behavior clearly. | Preserves exact behavior while fixing misleading positional argument usage. | Verified identical component count (89 components). |

#### 4. Vessel Segmentation Preprocessing (`retina_grayscale.py`)
| Target | Upstream behavior | Modified behavior | Rationale & Citation | Verification |
|:---|:---|:---|:---|:---|
| **Out-of-FOV padding** (`_mean_value`) | Indexes image with an integer mask `image[mask] = value`, which overwrites row 0 and row 1 of the image instead of out-of-FOV pixels. | Replaces out-of-FOV pixels (`mask == 0`, matching `restore_mask`) with the in-FOV mean intensity. | **Marín 2011, p. 149:** Out-of-FOV pixels are padded with the mean gray level of the FOV before filtering to suppress step-edge artifacts at the FOV boundary. | Out-of-FOV region is uniformly filled with the FOV mean. |
| **Background homogenization** (`homogenize`) | Shifts intensities by `g + 180 - np.max(image)` (where `max` is almost always 255). | Shifts intensities by `g + 128 - mode(in-FOV g)` and clips to `[0, 255]`. | **Marín 2011, Eqs. (2)–(3):** $g_{\text{Input\_Max}}$ is the gray-level where the intensity histogram reaches its maximum (the mode), shifted to 128. | In-FOV histogram mode becomes 128. Bundled test image Dice: 0.7653 → 0.7664. |

#### 5. Artery/Vein Classification & Server (`vessel_classification.py`, `endpoint_vessel_classification.py`)
| Target | Upstream behavior | Modified behavior | Rationale & Citation | Verification |
|:---|:---|:---|:---|:---|
| **Color channel order in REST endpoint** | Decodes uploaded image via PIL (RGB) and passes it directly to `classification()`, which expects OpenCV BGR order. | Converts RGB → BGR before calling `classification()` and converts output BGR → RGB before returning. | Pretrained weights and upstream tests (`cv2.imread`) use BGR order. Passing RGB swaps Red and Blue channels. | Bundled image via endpoint: accuracy 0.548 (0 artery pixels) → matches direct BGR call (accuracy 0.886; 1046 red artery px, 713 blue vein px). |
| **Border padding in `classification()`** | Calls `potential_landmarks(0)` with zero padding, crashing on vessels near the image border. | Calls `potential_landmarks(20)` (matching the landmark endpoint) while returning coordinates in the original image frame. | Prevents out-of-bounds slice crashes on standard fundus images. | Runs without crashing on the bundled image. |
| **Model inference without Keras/TensorFlow** | Requires legacy Keras 2.2 / TensorFlow at runtime to load `model_weights.h5` and `model_architecture.json`. | Reads the HDF5 weights directly with `h5py` and computes the 3-layer MLP forward pass (Dense 27→40 ReLU → 40 ReLU → 2 Softmax) in pure NumPy. | Keras 2.2 JSON/HDF5 format fails to load in modern Keras 3 / TensorFlow 2.16+. | Exact match against upstream `loading_model_predictions_test.csv`. |
| **Precomputed feature file protection** | `preparing_data()` overwrites the bundled `training_data.h5` with an empty file if the DRIVE training folder is absent. | Returns early without overwriting `training_data.h5` when no training images are found. | Prevents running unit tests from corrupting the repository's bundled data file. | `training_data.h5` remains intact after running tests. |

#### 6. Other Bug Fixes & Modern Python Compatibility
- **`_curve_to_image` (`retina.py`):** Avoids mutating caller coordinate arrays in-place and fixes `uint8` subtraction overflow on Windows/NumPy 2.
- **`detect_vessel_border` (`retina.py`):** Copies input image before skeleton traversal so the caller's `Retina.np_image` is not wiped out.
- **`_create_tag_image` & `set_tag_layer` (`retina.py`):** Supports 7-element tag tuples produced by `evaluate_window` and appends tag layers instead of overwriting the image.
- **`math.py` bounds check:** Fixes inverted condition `len(y) - 1 <= target <= 0` (which was almost always false) to `len(y) - 2 < target or target < 0`.
- **`save_image` & `save_window` (`retina.py`):** Fixes path separator concatenation and converts floating-point arrays to `uint8` before saving with modern Pillow.
- **`compare_with` (`retina.py`):** Casts boolean masks to `int` before subtraction so binary comparisons do not fail under NumPy 2.
- **`original_base64` (`retina_grayscale.py`):** Scales `[0, 1]` float RGB images to `[0, 255]` `uint8` instead of truncating to all-zeros (black image).
- **NumPy 2 & `setup.py`:** Replaces deprecated `np.int`, `np.float`, and `np.trapz`; removes `tensorflow`, `keras`, and `scikit-learn` from `install_requires` and adds `opencv-python`.

### Unchanged Components (Preserved Intentionally)
- **Pretrained A/V model feature quirks (`_vessel_widths`, LBP sign convention, `_postprocessing` `width * 1.75`):** Although `_vessel_widths` has the same 1-px undercount as `landmarks.vessel_width` and the Local Binary Pattern uses `<` instead of Ojala's $\ge$, the bundled MLP weights (`model_weights.h5`) were trained on these exact feature values. Changing them would invalidate the pretrained classifier.
- **`pyrUp` / `pyrDown` no-op in `retina_grayscale.py`:** Upstream called `cv2.pyrUp(image)` and `cv2.pyrDown(image)` without assigning the return values. Actually applying pyramid scaling degrades segmentation Dice from 0.7664 to 0.5627 because subsequent thresholds were tuned without it; hence the unused calls were simply removed.
- **Distance measure tortuosity $+1$ offset (`distance_measure_tortuosity`):** Returns $\text{arc} / \text{chord}$ ($\ge 1$) rather than $\text{arc} / \text{chord} - 1$ (Hart 1999 $\tau_1$), preserving upstream scale compatibility.
- **Window mean weighting (`evaluate_window`):** Uses unweighted arithmetic mean across vessel segments in a window rather than Hart 1999's length-weighted mean, preserving upstream column semantics.

### Running the Tests
```bash
python run_tests.py
```
All 146 unit tests (including `retipy/test/test_fork.py`) pass on both modern environments (NumPy 2.x, SciPy 1.15+, scikit-image 0.26, OpenCV 4.x/5.x) and legacy environments (NumPy 1.23, scikit-image 0.19, OpenCV 4.7).

---

<a id="japanese"></a>
## 日本語: 改正点と根拠（引用文献）

### 改定の方針
1. **原著論文または数学の定義に照らして誤りと断定できるものを修正する。** 根拠となる文献・式番号を下の表に明記する。
2. **根拠がないものは変えない。** 原著が特定できない処理や、変更すると精度が悪化する処理は本家の動作を維持し、その理由を記録する。
3. **学習済みの重みが前提としている特徴量の定義は変えない。** 動静脈分類の重み（`model_weights.h5`）が学習時に前提とした特徴量定義を変えると重みが使えなくなるため維持する。
4. **真の値が分かる解析的な入力で検証する。** 本家のテストは出力を既存の CSV と比較するスナップショットテストが多く、数式の誤りを検出できない。そのため、真の値が既知の合成曲線・図形で検証する `retipy/test/test_fork.py` を追加した。

### 引用文献（根拠）
| 略称 | 文献情報 | 根拠として用いた箇所 |
|:---|:---|:---|
| **Hart 1999** | Hart WE, Goldbaum M, Côté B, Kube P, Nelson MR. Measurement and classification of retinal vascular tortuosity. *Int J Med Inform*. 1999;53(2-3):239–252. doi:10.1016/S1386-5056(98)00163-4 | 式(2)、p.244 表1および total squared curvature ($tsc$) の定義、p.245 の中心線の平滑化と弧長の定義 |
| **Grisan 2003** | Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic evaluation of retinal vessel tortuosity. *Proc 25th Annu Int Conf IEEE EMBS*. 2003;1:866–869. doi:10.1109/IEMBS.2003.1279902 | 式(13)〜(15)（弧長による媒介変数表示と積分）、曲率ヒステリシス閾値（静脈 0.01、動脈 0.03） |
| **Grisan 2008** | Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic grading of retinal vessel tortuosity. *IEEE Trans Med Imaging*. 2008;27(3):310–319. doi:10.1109/TMI.2007.904657 | 式(17)〜(19)、p.314（変曲点による分割と蛇行密度 Tortuosity Density $TD$ の定義） |
| **Marín 2011** | Marín D, Aquino A, Gegúndez-Arias ME, Bravo JM. A new supervised method for blood vessel segmentation in retinal images by using gray-level and moment invariants-based features. *IEEE Trans Med Imaging*. 2011;30(1):146–158. doi:10.1109/TMI.2010.2064333 | 式(1)〜(3)、p.149（視野外の平均値置換と背景均質化の最頻値シフト） |
| **Calvo 2011** | Calvo D, Ortega M, Penedo MG, Rouco J. Automatic detection and characterisation of retinal vessel tree bifurcations and crossovers in eye fundus images. *Comput Methods Programs Biomed*. 2011;103(1):28–38. doi:10.1016/j.cmpb.2010.06.002 | 式(3)の交差数（crossing number $CN(p)$）による骨格分岐点の判定 |
| **Wikipedia** | Minkowski–Bouligand dimension（ミンコフスキー・ブーリガン次元 / ボックスカウンティング次元） | フラクタル次元の定義（集合を覆うのに要する箱の総数 $N(\varepsilon)$）。本家のコードが挙げる典拠 |

※本家がコード内で明示的に引用しているのは Hart 1999、Grisan 2003、Wikipedia の3つである。Marín 2011 と Calvo 2011 はアルゴリズムの一致から特定した比較根拠である。

### 改正点の一覧

#### 1. 蛇行度（`tortuosity_measures.py`, `retina.py`, `tortuosity.py`, `math.py`）
| 対象 | 本家の実装 | この版の実装 | 根拠・引用 | 検証結果（本家 → この版） |
|:---|:---|:---|:---|:---|
| **点の順序** `detect_vessel_border` | $x$ 座標で並べ替えて1行1点に間引き、血管木全体を1本の曲線として扱う | 分岐点（Calvo 2011 式(3)の交差数 $\ge 3$）で区切り、各区間を端点から血管に沿った順にたどる。閉じた輪は返さない | **Hart 1999, p.245：** 弧長は曲線に沿って隣り合う点の距離の和。$x$ 順のソートでは水平な血管が1点に潰れ、分岐木が折れ線になる | 水平な261画素：1点 → 261点。<br>正弦波（真の値 1.582）：縦 1.625、横 1.069 → 縦横とも 1.573。<br>半円（真の値 1.571）：1.576、1.143 → 縦横とも 1.593 |
| **曲線の表現** | 画素の離散点列をそのまま有限差分 | 平滑化スプライン（`Centerline`, `splprep`, $s=0.35N$）を当ててから弧長と曲率を計算 | **Hart 1999, p.245; Grisan 2003, 2008：** 画素の階段を直接微分すると離散化雑音で曲率が埋もれる | 下の各行で検証 |
| **二乗曲率積分** `squared_curvature_tortuosity` | $\left|\int \kappa \, dt\right|$（二乗せず符号つきのまま積分して絶対値をとるため、逆向きの曲がりが相殺する） | $\int \kappa(s)^2 \, ds$（曲率の二乗を弧長 $s$ で積分） | **Hart 1999, p.244 表1の $tsc$。** 弧長で積分する形は **Grisan 2003 式(15)** | 対称なS字（真の値 0.1461）：0.0000 → 0.1434。<br>円弧（真の値 0.0149）：0.3412 → 0.0150 |
| **2階差分** `derivative2_centered_h1` | $(y_{i+1} - 2y_i + y_{i-1}) / 4$ | 4で割らない（間隔 $h=1$ の中心2階差分） | 間隔 $h=1$ の2階中心差分の数学的定義（分母は $h^2 = 1$） | $y = x^2$ の2階微分（真の値 2）：0.5 → 2 |
| **変曲点** `_detect_inflection_points` | $y$ 座標の増減の反転を数える | スプラインの曲率 $\kappa(s)$ の符号の変化を検出し、ヒステリシス閾値（0.01）を超えた区間の間の中点で分ける | **Grisan 2008 式(17) と p.314、Grisan 2003：** 変曲点は曲率 $\kappa$ の符号が変わる点であり、$y$ の極値ではない | 傾き 1/2 の直線（真の値 0）：58 → 0。<br>ラスタライズした正弦波（真の値 1）：56 → 1 |
| **蛇行密度** `tortuosity_density` | $\frac{n-1}{n} + \frac{1}{L}\sum\dots$（$+$ になっており、$n$ に変曲点数を入れ、最後の区間と各変曲点直前の区間が欠落） | $\frac{n-1}{n} \times \frac{1}{L}\sum_{i=1}^{n}\left(\frac{L_{cs_i}}{L_{xs_i}} - 1\right)$（$n$ は曲率の符号が一定の部分区間の数で、全 $n$ 区間を足す） | **Grisan 2008 式(18)(19)** | 傾き 1/2 の直線：0.9828 → 0。<br>正弦波：式(19)の理論値 0.00351 に対して 0.00346 |
| **窓の閾値判定** `tortuosity.density`, `fractal` | 既定の閾値 `0.97`, `0.94` 未満の窓を捨てる | 既定を `None`（全ての窓を返す）にし、出力の有効桁数を増やした | `0.97` は誤った加算式の値域に合わせた値であり、原著の根拠がない | 本家では傾き 1/2 の直線が「蛇行」と判定され正弦波が検出されなかった → 直線 0.00000、正弦波 最大 0.0049 |
| **決定係数 $R^2$** `linear_regression_tortuosity` | 説明平方和 $\div$ 全平方和（$\text{SSR}/\text{SST}$） | $1 - \text{残差平方和} \div \text{全平方和}$（$1 - \text{SSE}/\text{SST}$） | **決定係数の定義：** 最小二乗回帰直線ではない2端点を結ぶ弦では $\text{SST} = \text{SSR} + \text{SSE}$ が成り立たず、前者は 1 を大きく超える | ジグザグ：361 → −367。<br>半円：3.14 → −3.14。<br>直線：1 のまま |
| **窓内の平均** `evaluate_window` | `t4` と `tfi` だけ血管数で割らず合計のまま出力 | 全ての指標を血管数 `vessel_count` で割る | 他の列と同じく窓内の血管1本あたりの平均にするため | 同じ血管を2本にしたとき：`t4` が 0.228 → 0.456、`tfi` が 0.935 → 1.870 に倍増していたものが、それぞれ 0.0053、0.9305 で一定 |
| **未完成の関数** `smooth_tortuosity_cubic` | `y[0]` を返すだけの未実装関数 | 削除 | 動作しない死蔵コード | — |

#### 2. フラクタル次元（`retipy/lib/fractal_dimension.py`）
| 対象 | 本家の実装 | この版の実装 | 根拠・引用 | 検証結果 |
|:---|:---|:---|:---|:---|
| `fractal_dimension` | 全てが埋まった箱（$S = k^2$）を数えず境界の箱だけを数える。また画素値の範囲をそのまま閾値に使う | 0 を超える画素を前景集合とし、前景を1画素でも含む全ての箱（$S > 0$）を数える | **Minkowski–Bouligand 次元の定義：** 集合を覆うのに必要な箱の総数 $N(\varepsilon)$ を数える | 塗りつぶした円：1.148 → 1.750。<br>0/255 の二値画像：`NaN` → 1.750。<br>細い線：1.082 のまま |

#### 3. 分岐と交叉（`landmarks.py`）
| 対象 | 本家の実装 | この版の実装 | 根拠・引用 | 検証結果 |
|:---|:---|:---|:---|:---|
| **交叉の箱の座標** | 右下の座標だけ余白 `border` の符号が逆（`+ border`） | 分岐と同じく `- border` に統一 | パディング除去時の座標計算の誤り（同じ関数の分岐の行と不一致） | 余白 0, 2, 20 で箱の幅が 6, 10, 46 → すべて 6 |
| **血管幅** `vessel_width` | 反対側の1画素目を調べず、中心画素も数えないため 1 画素少ない | 中心画素を含め、両側の和が実際の血管幅と一致するように修正 | 幅の幾何学的定義 | 幅 1, 3, 5, 7, 9 画素：1, 2, 4, 6, 8 → 1, 3, 5, 7, 9 |
| **画像の端の処理** | 負の添字で反対側の端を読む。空の領域を OpenCV に渡してクラッシュする | 画像外は背景（0）として扱い、空領域のガードを追加 | 配列の折り返し参照と OpenCV 4.7 以降でのクラッシュ防止 | 余白 0 での異常終了が解消 |
| **連結性の指定** | `connectedComponentsWithStats(img, 4, CV_32S)` で第2引数 `labels` に 4 を渡しており、実際は既定の8連結で動作 | `connectivity=8` とキーワード引数で明記 | 実際の動作を変えずに誤った位置引数を正す | 成分数（89成分）が一致することを確認 |

#### 4. 血管分割の前処理（`retina_grayscale.py`）
| 対象 | 本家の実装 | この版の実装 | 根拠・引用 | 検証結果 |
|:---|:---|:---|:---|:---|
| **視野外の置換** `_mean_value` | マスクが整数配列のため `image[mask] = value` が先頭2行を塗る動作になっていた | 視野外（`mask == 0`、`restore_mask` と同じ条件）を視野内の平均値で置き換える | **Marín 2011, p.149：** 視野境界のステップエッジを抑えるため視野外を視野内平均濃度で埋める | 視野外がすべて平均値で塗られる |
| **背景均質化** `homogenize` | `g + 180 - np.max(image)`（最大値はほぼ常に 255） | `g + 128 - 視野内の最頻値` | **Marín 2011, 式(2)(3)：** $g_{\text{Input\_Max}}$ はヒストグラムの画素数が最大となる濃度（最頻値）であり、これを 128 にシフトする | 視野内の最頻値が 128 になる。同梱画像の Dice は 0.7653 → 0.7664 |

#### 5. 動静脈分類とサーバー（`vessel_classification.py`, `endpoint_vessel_classification.py`）
| 対象 | 本家の実装 | この版の実装 | 根拠・引用 | 検証結果 |
|:---|:---|:---|:---|:---|
| **REST サーバーの色順** | PIL で読んだ RGB 画像をそのまま `classification()` に渡す | BGR に変換して渡し、出力画像を RGB に戻して返す | 学習データと本家のテストは OpenCV の BGR 順であり、RGB のまま渡すと赤と青が入れ替わる | サーバー経由の正解率 0.548（動脈 0 画素）→ BGR 直接呼び出しと同じ結果（正解率 0.886、動脈赤 1046 画素・静脈青 713 画素） |
| **分類時の余白** | `potential_landmarks(0)`（余白 0）で呼び、端の血管で止まる | 余白 20 で呼び、元の画像座標系で返す | 分岐検出エンドポイントと同じ余白を与え、端でのクラッシュを防ぐ | 同梱画像で停止せず完走 |
| **Keras 依存の解消** | 実行時に古い Keras 2.2 / TensorFlow が必須 | `h5py` で重みを読込み、NumPy のみで順伝播を計算 | Keras 2.2 形式の JSON/H5 は現在の Keras 3 では読み込めないため | 本家の予測値 `loading_model_predictions_test.csv` と完全一致（`test_loading_model`） |
| **同梱の特徴量ファイルの保護** | 学習用画像フォルダがない状態で `preparing_data()` を呼ぶと同梱の `training_data.h5` を空で上書きする | 学習用画像が見つからない場合は上書きせず終了する | テスト実行によってリポジトリ同梱の H5 ファイルが破壊されるのを防ぐ | テスト実行後も `training_data.h5` が維持される |

#### 6. その他の修正・環境対応
- **`_curve_to_image`：** 呼び出し元の座標配列を書き換えないようにし、Windows / NumPy 2 での `uint8` 引き算のオーバーフローを解消。
- **`detect_vessel_border`：** 入力画像をコピーして処理し、呼び出し元の画像を破壊しないように修正。
- **`_create_tag_image`, `set_tag_layer`：** `evaluate_window` が出力する7要素のタグで止まらないようにし、画像を上書きせずレイヤーを追加するよう修正。
- **`math.py` の範囲確認：** `len(y) - 1 <= target <= 0` が常に偽になっていた誤りを `len(y) - 2 < target or target < 0` に修正。
- **`save_image`, `save_window`：** パス区切りの欠落を修正し、新しい Pillow でも浮動小数点画像を保存できるよう修正。
- **`compare_with`：** 二値（bool）画像同士の引き算でエラーにならないよう修正。
- **`original_base64`：** RGB の浮動小数点画像（`[0, 1]`）が整数変換で真っ黒（0）になっていた不具合を修正。
- **NumPy 2 対応・依存パッケージ整理：** 削除された `np.int`, `np.float`, `np.trapz` に対応し、`setup.py` から実行に不要な `tensorflow`, `keras`, `scikit-learn` を外して `opencv-python` を追加。

### 変更していない点（意図的に維持した仕様）
- **学習済みモデルの前提となっている特徴量（`_vessel_widths` の1画素不足、LBP の不等号の向き、`_postprocessing` の `width * 1.75`）：** 動静脈分類の学習済み重み（`model_weights.h5`）はこれらの特徴量定義のもとで学習されているため、変更すると重みが無効になる。
- **`pyrUp` と `pyrDown`（呼び出しの削除のみ）：** 本家は戻り値を変数に代入しておらず、実際には拡大も縮小も行われていなかった。実際に拡大・縮小を適用すると後続の閾値と合わず Dice が 0.7664 から 0.5627 に低下するため、元の動作を維持して無効な呼び出しのみ削除した。
- **距離法蛇行度の $-1$ の有無（`distance_measure_tortuosity`）：** Hart 1999 の $\tau_1$ は $\text{弧長}/\text{弦長} - 1$ だが、本家の出力スケールに合わせて $\text{弧長}/\text{弦長}$ のままとした（順位や標準偏差あたりのオッズ比は不変）。
- **窓内の血管平均の重みづけ（`evaluate_window`）：** Hart 1999 (p.242–243) は血管長による重み付き平均を推奨しているが、本家の仕様に合わせて単純平均のままとした。

### テストの実行方法
```bash
python run_tests.py
```
新規追加した `retipy/test/test_fork.py` を含む全 146 件のテストが、最新環境（NumPy 2.x, SciPy 1.15+, scikit-image 0.26, OpenCV 4.x/5.x）および旧環境（NumPy 1.23, scikit-image 0.19, OpenCV 4.7）の双方で通過することを確認済みである。

---

<a id="original-readme"></a>
## Original README

[![Build Status](https://travis-ci.org/alevalv/retipy-python.svg?branch=master)](https://travis-ci.org/alevalv/retipy-python)
[![Coverage Status](https://codecov.io/gh/alevalv/retipy-python/branch/master/graph/badge.svg)](https://codecov.io/gh/alevalv/retipy-python)

retipy-python is part of the [retipy](https://github.com/alevalv/retipy) project.

The goal of this project is to create a python library to perform different image processing operations on fundus retinal images. Currently there are vessel segmentation, bifurcation detection and tortuosity measures available as a REST endpoints.

### Installation

#### Development Environment

This project uses [OpenCV](https://opencv.org/) 4.0.0. Any version older than 4 will not work.

To use this project locally and be able to make changes to the retipy code, you can run the following command in
your console (having python3 and pip installed):

```bash
pip install --user -e .
```

This command should be ran inside the src folder that contains the retipy folder. It will make the retipy
library available to the user that ran it.

#### Docker

The library is also available as a docker container at [alevalv/retipy-python](https://hub.docker.com/r/alevalv/retipy-python/):

```bash
docker pull alevalv/retipy-python:latest
```
By default, the docker image will expose a REST endpoint in port 5000.

### License
retipy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
