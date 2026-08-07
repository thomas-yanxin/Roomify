<div align="center">

# Roomify

**把户型图变成可计算的 JSON。**

Roomify 读取住宅户型图或 PDF，输出房间、墙体、门窗和毫米级几何数据。

[English](README.md) · [简体中文](README_ZH.md)

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![状态：Beta](https://img.shields.io/badge/status-beta-6f42c1)
![许可证：Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2ea44f)

</div>

OpenCV 负责测量几何，视觉语言模型负责读取文字和符号。所有坐标都来自图像
计算，不由模型生成。没有配置 VLM 时，Roomify 仍会输出像素几何，并明确列出
无法确定的字段。

```bash
roomify floorplan.png -o floorplan.json
```

## 示例

仓库内包含这张真实户型图：

<p align="center">
  <img src="examples/floorplan-1.png" alt="Roomify 示例使用的住宅户型图" width="620">
</p>

[`floorplan-1.png`](examples/floorplan-1.png) →
[`floorplan-1.json`](examples/floorplan-1.json)

Roomify 从图中得到：

- 9 个房间
- 55 段墙体
- 18 个墙体开口
- 分别计算横向与纵向比例尺，置信度为 `high`
- 客厅计算面积为 37.34 m²，图纸标注为 37.52 m²

完整结果中的部分字段如下：

```json
{
  "schema_version": "1.0",
  "scale": {
    "px_per_mm_x": 0.0431736218444101,
    "px_per_mm_y": 0.048002053563788824,
    "method": "dimension_chains+printed_areas",
    "confidence": "high"
  },
  "rooms": [
    {
      "id": "room_1",
      "name": "客厅",
      "room_type": "living_room",
      "area_sqm": 37.33987625081361,
      "printed_area_sqm": 37.52,
      "area_deviation_flag": false,
      "source": "cv+vlm",
      "confidence": 0.95
    }
  ],
  "openings": [
    {
      "id": "op_1",
      "element_type": "window",
      "width_mm": 1250.763723150358,
      "wall_id": "wall_43",
      "connects": ["room_4", "exterior"],
      "source": "cv+vlm",
      "confidence": 0.9
    }
  ]
}
```

## 安装

Roomify 需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/thomas-yanxin/Roomify.git
cd Roomify
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

项目暂未发布到 PyPI。请使用干净的虚拟环境：不同 OpenCV wheel 都提供
`cv2`，不能混装。

## 快速开始

下面展示的语义和毫米数据需要先配置下一节中的 VLM 环境变量。

### 命令行

```bash
roomify examples/floorplan-1.png -o plan.json
```

支持常见图片格式和 PDF。`--page N` 用于选择 PDF 页，`--no-vlm` 只运行
计算机视觉部分，`--debug DIR` 保存各阶段的掩码与标记图。

### Python

```python
from roomify import parse

plan = parse("examples/floorplan-1.png")

print(plan.rooms[0].name)       # 客厅
print(plan.rooms[0].area_sqm)   # 37.33987625081361
print(plan.scale.confidence)    # high

plan.model_dump_json(indent=2)
```

如果只需要像素几何，不调用 VLM：

```python
plan = parse("floorplan.png", use_vlm=False)
```

## 配置 VLM

Roomify 支持能够接收图片的 OpenAI 兼容接口。凭据从进程环境中读取：

```bash
export ROOMIFY_VLM_API_KEY="..."
export ROOMIFY_VLM_BASE_URL="https://your-endpoint.example/v1"
export ROOMIFY_VLM_MODEL="your-vision-model"
```

Roomify 不会自行加载 `.env` 文件。`.env.example` 列出了全部环境变量。

## JSON 结构

每次解析都会返回经过校验的 `FloorPlan` 文档：

```text
FloorPlan
├── source_file, source_sha256, page
├── image_width_px, image_height_px, north_angle_deg
├── scale
│   ├── px_per_mm_x, px_per_mm_y, anisotropy
│   └── method, confidence, 证据数量
├── rooms[]
│   ├── name, room_type, source, confidence
│   ├── polygon_px, area_px, perimeter_px, edge_lengths_px
│   ├── polygon_mm, area_sqm, perimeter_mm, edge_lengths_mm
│   └── printed_area_sqm, area_deviation, area_deviation_flag
├── walls[]
│   ├── start_px, end_px, thickness_px
│   ├── start_mm, end_mm, thickness_mm
│   └── rooms
├── openings[]
│   ├── element_type, bbox_px, center_px, width_px, width_mm
│   ├── wall_id, connects, swing, hinge_px
│   └── source, confidence
├── elements[]
├── warnings[]
└── unresolved[]
```

主要约定：

- 像素坐标使用原始输入图片。原点在左上角，x 向右，y 向下。
- 多边形使用开放环，末尾不会重复第一个点。
- `scale` 为 `null` 时，所有毫米和平方米字段也为 `null`。
- `warnings` 记录降级和证据冲突。
- `unresolved` 列出无法从图纸中确定的字段。

完整字段和枚举定义见 [`schema.py`](src/roomify/schema.py)。

## 工作方式

1. 读取图片或 PDF，并保留到原始像素坐标的映射。
2. 使用计算机视觉检测墙体、封闭房间和墙体开口。
3. 将带编号的标记图发送给 VLM，读取房间名称、标注面积、尺寸和符号类型。
4. 合并两类结果，分别计算 x/y 比例尺，再用 Pydantic 校验输出。

几何计算不依赖 VLM 成功返回。调用失败时，Roomify 会保留几何结果，写入警告，
并把相关字段标为未解析。

## 当前范围

- 实心填充墙体的住宅户型图效果最好。
- 纯线框 CAD 图会走较简单的回退路径。
- 两个空间之间完全没有分隔线时，会被识别为一个房间。
- 暂不测量飘窗向外凸出的多边形。
- 当前版本用房间内侧边界近似外墙中心线。

## 开发

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src
```

在线 VLM 验收测试使用仓库内的示例，并需要配置三个 VLM 环境变量：

```bash
uv run pytest tests/integration
```

## 许可证

[Apache-2.0](LICENSE)
