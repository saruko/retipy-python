# retipy-python (Modified Fork)

[English](#english) | [日本語](#japanese)

---

<a id="english"></a>
## English: Detailed Revision Notes & References

This repository is a modified fork of [alevalv/retipy-python](https://github.com/alevalv/retipy-python) (commit `e6f0ed2`, 2019-05-18), revised on 2026-11-02 to match the definitions in the original papers.

### License
GPL-3.0 ([LICENSE.md](LICENSE.md)), unchanged from upstream. Every modified source file includes a header notice with the modification date. Keep this file and `LICENSE.md` together when redistributing.

### Revision Policy
1. **Fix errors verifiable against original papers or mathematical definitions.** Rationale and citations are listed in the tables below.
2. **Do not change code without clear evidence.** Components whose source literature is unknown or where changes degrade accuracy are kept as in upstream, with reasons documented.
3. **Preserve values assumed by pretrained weights.** Changing feature definitions used during training invalidates the bundled artery/vein classification weights.
4. **Verify results on inputs with known analytical ground truth.** Most upstream tests only compare outputs against recorded CSV files and cannot detect formula errors.

### References
| Short cite | Full citation | Used for |
|:---|:---|:---|
| **Hart 1999** | Hart WE, Goldbaum M, Côté B, Kube P, Nelson MR. Measurement and classification of retinal vascular tortuosity. *Int J Med Inform*. 1999;53(2-3):239–252. doi:10.1016/S1386-5056(98)00163-4 | Eq. (2), p. 244 Table 1 and definition of $tsc$, p. 245 smoothing |
| **Grisan 2003** | Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic evaluation of retinal vessel tortuosity. *Proc 25th Annu Int Conf IEEE EMBS*. 2003;1:866–869. doi:10.1109/IEMBS.2003.1279902 | Eqs. (13)–(15), curvature hysteresis values |
| **Grisan 2008** | Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic grading of retinal vessel tortuosity. *IEEE Trans Med Imaging*. 2008;27(3):310–319. doi:10.1109/TMI.2007.904657 | Eqs. (17)–(19) and p. 314 |
| **Marín 2011** | Marín D, Aquino A, Gegúndez-Arias ME, Bravo JM. A new supervised method for blood vessel segmentation in retinal images by using gray-level and moment invariants-based features. *IEEE Trans Med Imaging*. 2011;30(1):146–158. doi:10.1109/TMI.2010.2064333 | Eqs. (1)–(3), p. 149 out-of-FOV padding |
| **Calvo 2011** | Calvo D, Ortega M, Penedo MG, Rouco J. Automatic detection and characterisation of retinal vessel tree bifurcations and crossovers in eye fundus images. *Comput Methods Programs Biomed*. 2011;103(1):28–38. doi:10.1016/j.cmpb.2010.06.002 | Eq. (3) crossing number |
| **Wikipedia** | Minkowski–Bouligand dimension | Definition of box-counting fractal dimension (cited in upstream code) |

Upstream explicitly cites only Hart 1999, Grisan 2003, and Wikipedia. Vessel segmentation, landmark detection, and A/V classification have no citations in upstream code; Marín 2011 and Calvo 2011 are used where the preprocessing and crossing-number formulas correspond to those papers.

### Summary of Changes

See [README.md](README.md) for the full side-by-side tables of changes in English and Japanese.

#### A/V Classification Post-processing (`× 1.75`) and Vessel Width Ablation
Evaluated via `classification()` on the bundled test image (234×278):

| Vessel width | Post-processing rule | Accuracy | Sensitivity | Specificity |
|:---|:---|---:|---:|---:|
| None (raw MLP) | — | 0.884 | 0.957 | 0.831 |
| Upstream | `× 1.75` (upstream rule) | 0.899 | 0.968 | 0.845 |
| Upstream | Widest vessel | 0.883 | 0.972 | 0.820 |
| **Fixed (`landmarks.vessel_width`)** | **`× 1.75` (this fork)** | **0.886** | **0.965** | **0.829** |
| Fixed (`landmarks.vessel_width`) | Widest vessel | 0.903 | 0.965 | 0.859 |

#### Parameters
| Name | Value | Rationale |
|:---|---:|:---|
| `SMOOTHING` | 0.35 | Determined on synthetic curves (minimum value that eliminates spurious inflections on rasterized straight lines) |
| `HYSTERESIS` | 0.01 | Grisan 2003 threshold for veins (arteries: 0.03) |
| `SAMPLES_PER_PIXEL` | 2 | Spline sampling density per pixel of chord length |

#### Vessel Segment Splitting (`detect_vessel_border`)
`detect_vessel_border` (1) removes staircase corner pixels, (2) clears a 1-pixel neighborhood around junction pixels where Calvo's crossing number $CN(p) \ge 3$, and (3) traces the longest endpoint-to-endpoint path in each connected component. On the bundled manual mask and `img01.png`, 90.9% and 90.0% of skeleton pixels are retained in traced vessel segments (the remainder being junction neighborhood pixels).

#### Limitations
- Vessel segmentation and A/V classification were evaluated only on the single bundled test image (234×278).
- Tortuosity formulas were verified on synthetic curves with known ground truth; clinical validity on full-size fundus photographs is outside the scope of these unit tests.
- Original source papers for the full segmentation pipeline, landmark candidate rules, and A/V classification architecture remain unidentified.

---

<a id="japanese"></a>
## 日本語: 改定の詳細と根拠

[retipy-python](https://github.com/alevalv/retipy-python)（コミット `e6f0ed2`、2019-05-18）を、原著の定義に合うように直した版。2026-11-02 に改定した。

### ライセンス

GPL-3 のまま（[LICENSE.md](LICENSE.md)）。変更したファイルの冒頭に、変更した旨と日付を書いた。配布するときは、このファイルとライセンスを一緒に残す。

### 改定の方針

1. **原著か数学の定義で誤りと言えるものを直す。** 根拠は下の表に書いた。
2. **根拠がないものは変えない。** 原著が分からないもの、直すと悪くなるものは本家のまま残し、理由を書いた。
3. **学習済みの重みが前提にしている値は変えない。** 変えると動静脈分類の重みが使えなくなる。
4. **直した結果を、真の値が分かる入力で確かめる。** 本家のテストは出力を csv と比べるだけのものが多く、式の誤りを検出できない。

### 使った原著

| 略称 | 文献 | 使った箇所 |
|:---|:---|:---|
| Hart 1999 | *Int J Med Inform* 53:239–252, DOI 10.1016/S1386-5056(98)00163-4 | 式(2)、p.244 の表1と tsc の定義、p.245 の平滑化 |
| Grisan 2003 | *IEEE EMBC*, DOI 10.1109/IEMBS.2003.1279902 | 式(13)〜(15)、ヒステリシスの値 |
| Grisan 2008 | *IEEE TMI* 27:310–319, DOI 10.1109/TMI.2007.904657 | 式(17)〜(19) |
| Marín 2011 | *IEEE TMI* 30:146–158, DOI 10.1109/TMI.2010.2064333 | 式(1)〜(3)、p.149 の視野外の置換 |
| Calvo 2011 | *Comput Methods Programs Biomed* 103:28–38, DOI 10.1016/j.cmpb.2010.06.002 | 式(3)の交差数 |
| Wikipedia | Minkowski–Bouligand dimension | フラクタル次元の定義。本家のコードが挙げる典拠 |

本家が引用しているのは Hart、Grisan 2003、Wikipedia だけである。血管分割、分岐検出、動静脈分類には引用がない。Marín と Calvo は「この論文に基づくとすれば」の比較で、由来が確定したわけではない。

### 改定の内容

#### 蛇行度

| 対象 | 本家 | この版 | 根拠 | 確かめた結果（本家 → この版） |
|:---|:---|:---|:---|:---|
| 点の順序 `detect_vessel_border` | x で並べ替えて1行1点に間引く。木全体を1本として扱う | 分岐点で区切り、端点から血管に沿った順に返す。輪は返さない | Hart p.245：弧長は曲線に沿って隣り合う点の距離の和 | 水平な261画素：1点 → 261点。正弦波（真の値 1.582）：縦 1.625、横 1.069 → どちらも 1.573。半円（真の値 1.571）：1.576、1.143 → どちらも 1.593 |
| 曲線 | 画素の点列をそのまま微分 | 平滑化スプラインを当ててから計算（`Centerline`） | Hart p.245、Grisan 2003, 2008 | 下の各行 |
| 曲率 `squared_curvature_tortuosity` | `|∫κdt|`。二乗せず、符号つき | `∫κ²ds` | Hart p.244 の tsc。弧長で積分するのは Grisan 2003 の式(15)の形 | S字（真の値 0.1461）：0.0000 → 0.1434。弧（真の値 0.0149）：0.3412 → 0.0150 |
| 2階差分 `derivative2_centered_h1` | `/4` | 割らない | 間隔1の2階差分の定義 | x² の2階微分（真の値2）：0.5 → 2 |
| 変曲点 `_detect_inflection_points` | y の増減の変化 | スプラインの曲率の符号の変化。しきい値を超えた区間の間の中点で分ける | Grisan 2008 の式(17)と p.314 | 傾き1/2の直線（真の値0）：58 → 0。画素の正弦波（真の値1）：56 → 1 |
| 蛇行密度 `tortuosity_density` | `(n-1)/n + (1/L)Σ…`。n は変曲点の数で、最後の区間と各変曲点の直前の区間が入らない | `(n-1)/n × (1/L)Σ…`。n は符号が一定の部分の数で、全ての部分を足す | Grisan 2008 の式(18)(19) | 傾き1/2の直線：0.9828 → 0。正弦波：式(19)の値 0.00351 に対して 0.00346 |
| 窓の判定 `tortuosity.density`, `fractal` | 既定のしきい値 0.97, 0.94 | 既定は `None`（全ての窓を返す）。出力の桁も増やした | 0.97 は誤った式の値域に合わせた値で、原著の値ではない | 傾き1/2の直線が「蛇行」、正弦波が検出なし → 直線 0.00000、正弦波 最大 0.0049 |
| 「R²」 `linear_regression_tortuosity` | 説明平方和 ÷ 全平方和 | 1 − 残差平方和 ÷ 全平方和 | 決定係数の定義。回帰直線でない線に前者は使えない | ジグザグ：361 → −367。半円：3.14 → −3.14。直線：1 のまま |
| 平均 `evaluate_window` | t4 と tfi だけ血管数で割らない | すべて割る | 他の列と同じ平均 | 同じ血管を2本に：t4 0.228 → 0.456 だったものが 0.0053 で一定。tfi 0.935 → 1.870 だったものが 0.9305 で一定 |
| 未完成の関数 `smooth_tortuosity_cubic` | y[0] を返すだけ | 削除 | | |

#### フラクタル次元

| 対象 | 本家 | この版 | 根拠 | 確かめた結果 |
|:---|:---|:---|:---|:---|
| `fractal_dimension` | 全部が埋まった箱を数えない。値の範囲をそのまま使う | 集合を覆う全ての箱を数える。0 を超える値を集合とする | 本家が挙げる定義は「集合を覆うのに要する箱の数」 | 塗りつぶした円：1.148 → 1.750。0/255 の画像：NaN → 1.750。線：1.082 のまま |

#### 分岐と交叉（`landmarks`）

| 対象 | 本家 | この版 | 根拠 | 確かめた結果 |
|:---|:---|:---|:---|:---|
| 交叉の箱 | 右下の座標だけ余白の符号が逆 | 分岐と同じ符号 | 座標の計算の誤り。同じ関数の分岐の行と不一致 | 余白 0, 2, 20 で幅 6, 10, 46 → すべて 6 |
| `vessel_width` | 反対側の1画素目を調べず、中心も数えない | 反対側の値が中心を含み、和が血管幅になる | 幅の定義 | 幅 1, 3, 5, 7, 9：1, 2, 4, 6, 8 → 1, 3, 5, 7, 9 |
| 画像の端 | 負の番号で反対側を読む。空の領域を OpenCV に渡して止まる | 外は背景として扱う | | 余白0での停止（OpenCV 4.7）がなくなった |
| 連結の指定 | `connectedComponentsWithStats(img, 4, CV_32S)`。4 は連結性として効かず、実際は8連結 | `connectivity=8` と明記 | 動作は変わらない | 4連結を正しく指定すると 768 成分、実際の動作は 89 成分 |

#### 血管分割（`retina_grayscale`）

| 対象 | 本家 | この版 | 根拠 | 確かめた結果 |
|:---|:---|:---|:---|:---|
| 視野外の置換 | マスクが整数配列のため、先頭2行を平均値で塗る | 視野外を平均値で置き換える | Marín p.149。本家の `restore_mask` と同じ条件にした | 視野外がすべて平均値になる |
| `homogenize` | `g + 180 − 最大値`（最大値は常に255） | `g + 128 − 最頻値` | Marín の式(2)(3)。`g_Input_Max` は画素数が最も多い濃度 | 視野内の最頻値が 128 になる |

2つを合わせて、同梱の画像1枚（234×278）の Dice は 0.7653 → 0.7664 である。

#### 動静脈分類（`vessel_classification`）

| 対象 | 本家 | この版 | 根拠 | 確かめた結果 |
|:---|:---|:---|:---|:---|
| サーバーの色の順 | RGB をそのまま渡す | BGR に変えて渡し、出力を RGB に戻す | 学習データと本家のテストは BGR | 本家：正解率 0.548、動脈 0 画素 → サーバー経由で BGR の直接呼び出しと同じ結果（動脈は赤 1046 画素、静脈は青 713 画素） |
| 余白 | 余白0で呼び、端の血管で止まる | 余白20で呼ぶ。返る座標は元の画像のもの | サーバーの分岐検出と同じ値 | 同梱の画像で停止しなくなった |
| Keras | 読み込み時に必須 | 不要。同じ重みを NumPy で計算する | Keras 2.2 形式を現在の Keras で読めない | 本家の予測値の csv と一致（`test_loading_model`） |
| 同梱の特徴量ファイル | 学習用の画像がないと空で上書きする | 上書きしない | | 本家のテストを実行すると重みのフォルダの h5 が書き換わっていた |

**正解率は改善していない。** 同梱の画像1枚では、本家の式に血管幅の修正だけを入れると 0.899 → 0.886 である（下の表）。差は条件による揺れの範囲である。血管幅を直したのは、真の幅に合うことを確かめたからで、分類が良くなるからではない。

#### その他

| 対象 | 内容 |
|:---|:---|
| `_curve_to_image` | 呼び出し元の座標を書き換えない。Windows のオーバーフローを解消した |
| `detect_vessel_border` | 入力の画像を書き換えない |
| `_create_tag_image`, `set_tag_layer` | タグ7個（`evaluate_window` の出力）で止まらない。画像を上書きせず層を追加する |
| `math.py` の範囲の確認 | `len(y)-1 <= target <= 0` は常に偽に近く働かないので、正しい条件にした |
| `save_image`, `save_window` | パスの区切りを正した。新しい Pillow で小数の画像を保存できるようにした |
| `compare_with` | 二値の画像同士で止まらない |
| `original_base64` | RGB の画像で真っ黒になっていたのを直した |
| NumPy 2 など | `np.int`, `np.float`, `np.trapz` の削除に対応。`setup.py` から tensorflow, keras, scikit-learn を外し、opencv-python を加えた |

### 変更していない点

| 対象 | 理由 |
|:---|:---|
| **`_postprocessing` の `width*1.75`** | 名前は「最大幅」だが、最大とは限らない動きをする。一度「最大」に直して測ったが、結果は下の表のとおり上下し、誤りと言える根拠がない。原著も不明なので残した |
| **`pyrUp` と `pyrDown`（呼び出しは削除、動作は同じ）** | 本家は結果を捨てていて、拡大も縮小もしていなかった。実際に拡大と縮小をすると、Dice は 0.7664 → 0.5627 に下がった（縮小で生じた中間の濃度を、しきい値なしで血管と数えた値）。他の値がこの動作に合わせて決められているとみて、意図不明の呼び出しを外した |
| **LBP の向き** | Ojala の定義（周囲 ≥ 中心で1）と逆だが、重みがこの値で学習されている |
| **`_vessel_widths` の幅** | `landmarks.vessel_width` と同じ1画素の不足があるが、重みがこの値で学習されている |
| **半径 `w1 + w2*1.5`** | 括弧の付け忘れの疑いはあるが、原著が不明で断定できない。Calvo の半径は血管幅によらない固定値 |
| **候補点の 3×3 判定** | Calvo の交差数とは別の基準で、同梱の手動マスクでは候補173点中119点が分岐でない。直すと分類の特徴量が変わるので残した。ただし `detect_vessel_border` の区切りには交差数を使う |
| **Calvo との他の違い** | 枝の剪定、3種の半径の投票、組の解析がない。別設計であり誤りとは言えない |
| **Kondermann 2007 との違い** | 隠れ層40は同じだが、他はほとんど違う。別の手法である |
| **Marín との他の違い** | 背景の推定が中央値（Marín は 69×69 の平均）、σ=1.82（Marín は 1.8）、反射の除去の位置。この後の処理の由来が不明で、値の組み合わせを崩すと確かめられない |
| **Otsu 法以降の分割** | Marín の方法ではなく、由来が不明 |
| **距離法の −1** | Hart の τ1 は `s/chord − 1` だが、本家に合わせて −1 なしのまま。順位と SD あたりの OR は変わらない |
| **窓の中の平均** | Hart p.242–243 は長さの重みづけを求めるが、本家に合わせて単純平均のまま |
| **`_tortuosity_window` の x, y** | x に行番号が入っている。画面の縦横が入れ替わる可能性があるが、REST を通した確認ができていない |
| **FD の箱の大きさ** | 4画素以上だけを使う。本家と gist のまま |
| **Docker と Travis の設定** | tensorflow が残っている。実行を確認していないので触っていない |

#### `×1.75` と血管幅の組み合わせ

`classification()` と同じ経路、同梱の画像1枚。正解率、感度、特異度。

| 血管幅 | 規則 | 正解率 | 感度 | 特異度 |
|:---|:---|---:|---:|---:|
| 後処理なし | | 0.884 | 0.957 | 0.831 |
| 本家 | ×1.75（本家の式） | 0.899 | 0.968 | 0.845 |
| 本家 | 最も太い血管 | 0.883 | 0.972 | 0.820 |
| **修正後** | **×1.75（この版）** | **0.886** | **0.965** | **0.829** |
| 修正後 | 最も太い血管 | 0.903 | 0.965 | 0.859 |

### 設定値

| 名前 | 値 | 決め方 |
|:---|---:|:---|
| `SMOOTHING` | 0.35 | 合成の曲線で決めた値 |
| `HYSTERESIS` | 0.01 | Grisan 2003 の静脈の値（動脈は 0.03） |
| `SAMPLES_PER_PIXEL` | 2 | スプラインの標本の密度 |

### 血管の区切り方について

`detect_vessel_border` は、(1) 階段の角の余分な画素を除き、(2) Calvo の交差数が3以上の点の周囲を消して区切り、(3) 各区間で最も離れた2点を結ぶ経路をたどる。

返された経路に残る骨格の割合は、同梱の手動マスクで 90.9%、`img01.png` で 90.0% だった。残りはほぼ分岐点の周囲で消した画素である。

### テスト

```bash
python run_tests.py
```

146 件がすべて通ることを、次の2つの環境で確かめた。

| 環境 | numpy | scikit-image | OpenCV |
|:---|---:|---:|---:|
| 新しい環境 | 2.5.4 | 0.26.0 | 5.0.0 |
| 古い環境 | 1.23.5 | 0.19.3 | 4.7.0 |

- `retipy/test/test_fork.py`（新規）は、真の値や原著の式と比べる。
- 本家のテストのうち、意図した変更で値が変わるものは書き換えた。**csv の期待値は修正後の出力で作り直したもので、それ自体は正しさの証拠ではない。** 対象は `average_width_test`, `box_labels_test`, `boxes_bifurcations_test`, `boxes_crossings_test`, `finding_landmark_vessels_test`, `vessel_number_filter_test`, `vessel_number_skeleton_test`, `vessel_widths_test` と、`test_accuracy`, `test_calculate_roc` の値である。
- サーバーの4つの API は、Flask のテストクライアントで動くことを別に確かめた（`retipyserver/test` は実行していない）。

### 限界

- 血管分割と動静脈分類の評価は、同梱の小さな画像1枚（234×278）だけである。
- 蛇行度の確認は合成の曲線だけで、実際の眼底写真での妥当性は確かめていない。
- 血管分割、分岐検出、動静脈分類の原著は特定できていない。
