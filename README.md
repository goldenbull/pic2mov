# 家庭合影动画

这个项目用于把按月份保存的家庭合影制作成带渐变过渡的 MP4 动画。

`detect_faces.py` 负责检测人脸并生成供人工检查的结果；`make_animation.py`
读取检测结果和人工修正配置，固定 `person_2` 的脸部位置，并生成最终动画。

## 使用 uv 安装环境

```bash
uv sync
```

`uv sync` 会在项目目录中创建并管理 `.venv`，并安装 `pyproject.toml` 中声明的
依赖。如果需要进入交互式虚拟环境，可以执行：

```bash
source .venv/bin/activate
```

第一次运行人脸检测时，程序会把 YuNet 和 SFace 的 ONNX 模型下载到 `models/`。

## 执行人脸检测

```bash
uv run detect_faces.py \
  --input-dir /Users/cjn/Downloads/family_memories \
  --reference-image /Users/cjn/Downloads/family_memories/20180529.jpg \
  --output-dir face_review
```

`requirements.txt` 仍然保留给不使用 uv 的工具使用；本项目推荐使用
`uv sync` 和 `uv run`。

检测完成后，`face_review/` 中会包含：

* `detections.json`：所有候选人脸的位置、关键点、身份标签和相似度；
* `annotated/`：带有彩色标注的人脸预览图；
* `manual_config.json`：用于人工修正识别结果的配置模板。

终端输出中的 `candidate face(s)` 是检测到的全部候选人脸，可能包括其他家庭成员
或背景误检；`target=x/3` 只统计与参考图中三个人匹配的人脸。`review=n` 表示有
低相似度匹配，需要人工检查。原始候选数量会保存在 `detections.json` 的
`raw_face_count` 字段中。

如果需要调整检测参数，可以使用：

```bash
uv run detect_faces.py ... \
  --score-threshold 0.85 \
  --max-detect-size 1600 \
  --min-face-size 48 \
  --identity-threshold 0.30
```

参考图中的三张脸会从左到右标记为 `person_1`、`person_2`、`person_3`。检查预览
图后，可以在 `manual_config.json` 中修正漏检、多人合影或身份匹配错误的照片，
再通过 `--manual-config` 传给检测程序。被选中的人脸会被标记为有效，其他候选
人脸会从目标选择中排除。

人脸索引从 0 开始，顺序与该图片在 `detections.json` 中的 `faces` 数组一致。例如：

```json
{
  "overrides": {
    "20250129.jpg": {
      "use_face_indices": [0, 1, 2],
      "labels_in_face_order": ["person_2", "person_1", "person_3"]
    }
  }
}
```

如果问题只是图片末尾存在两个 `identity: null` 的候选人脸（例如
`20250129.jpg`），不需要手工删除。它们已经不会计入 `target_face_count`，而且重新
运行检测时直接修改 `detections.json` 也会被覆盖。

## 推荐的检查顺序

1. 把 `annotated/` 制作为联系表，或重点查看相似度较低的预览图。
2. 在 `manual_config.json` 中确认父亲、孩子和母亲分别对应哪个标签。
3. 为包含祖父母、其他人物或只有孩子的早期照片添加修正配置。
4. 使用修正后的配置生成动画。

## 生成动画

检查并修正 `detections.json` 和 `manual_config.json` 后，执行：

```bash
uv run make_animation.py \
  --detections face_review/detections.json \
  --manual-config face_review/manual_config.json \
  --output family_memory.mp4 \
  --width 1920 \
  --height 1080 \
  --hold-seconds 1.5 \
  --transition-seconds 0.5 \
  --background blur \
  --transform person2
```

默认的 `--transform person2` 只根据 `person_2` 进行旋转和等比例缩放，然后平移
整张图片，使 `person_2` 的脸保持在固定位置；另外两个人的位置不会参与对齐。
默认位置是画面水平居中、垂直方向位于输出高度的 `2/3`，也就是从底部向上约
三分之一处。可以使用 `--person2-x` 和 `--person2-y` 指定其他位置。

另外还提供两种模式：

* `--transform affine`：精确固定三个人脸的位置，但可能会拉伸背景；
* `--transform similarity`：减少背景拉伸，但允许有少量位置误差。

竖幅图片默认使用模糊背景补齐画面，也可以通过 `--background black` 或
`--background white` 使用纯色背景。增大 `--face-size-ratio` 可以让
`person_2` 在输出中显示得更大。

每张图片右上角会显示从文件名前 8 位数字解析出的月份。例如
`20230429.jpg` 会显示为红色的 `2023.04`。可以通过 `--date-font-scale` 调整
日期文字大小；日期字幕会和图片一起渐变。

渲染过程中，终端会显示当前文件名和总进度，例如：

```text
[001/101]   0.99% 处理 20180529.jpg
```

缺少所需 `person_2` 的图片会被跳过，并记录在 MP4 旁边的 `.report.json` 中。
该报告还会保存实际使用的图片顺序和所有动画参数。使用
`--preview-dir aligned_preview` 可以额外保存对齐后的静态图片，方便检查视频效果。
