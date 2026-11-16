# 3Dhex — STEP 实体晶格参数化填充工具

## 当前设计案例

最新版本为双边波浪齿晶格零件：两侧齿深各 20 mm，波浪侧壁参与开孔，
BCC 单胞 6 mm、杆径 1.0 mm，保留 35 道名义厚度 0.8 mm 的横向竖板。
中央 15 mm 通缝和右侧 0.5 mm 细缝保留。

- [设计参数、预览与复现步骤](LATEST_DESIGN.md)
- [下载打印模型（3MF / STL 压缩包）](https://github.com/maolin2501/3dhex/releases/tag/waist-open-wave-2026-11-16)
- [双边波浪齿 STEP 源模型](waist_bilateral_dense_wave_20261113/bilateral_dense_wave_20mm.step)

仓库包含软件源码、测试、CAD 输入和案例脚本；大型生成网格在 Releases 中下载。


## 可视化版（新增）

双击 `Start_Lattice_Studio.cmd` 打开本地三维设计软件。支持 STEP 原始 CAD 面选择、
局部晶格设计、指定外观面保留、11 种晶格（含 Kelvin 立体蜂窝）、单元大小和线径设置、
三维预览/剖切、进程停止及 STL 导出。完整说明见 [LATTICE_STUDIO.md](LATTICE_STUDIO.md)。
新增“顶点专用连接”：外层杆只接多边形顶点，不接孔边中部；右肩参考预设已使用此模式。
采用顶点内向短柱、内部直杆和相邻顶点内侧连接，适用于 BCC + 自动连接的轴向曲壳与近平面。
旧直杆接孔边、不规则孔网和固定孔型保留为对照，验证记录见 [VERTEX_VERIFICATION.md](VERTEX_VERIFICATION.md)。

下文为原命令行版说明；它的全局皮壳选项与可视化版“只保留所选面”不同。

输入 STEP 模型，选取其中的实体，**保留内外表面皮壳**，内部按参数化晶格填充，
输出可直接打印/仿真的水密网格（STL/OBJ/PLY/GLB）。

采用隐式建模流水线（nTopology 式）：体素化符号距离场 (SDF) + 周期隐式晶格函数 +
Marching Cubes 提取，无需昂贵的 B-rep 布尔运算，对复杂零件稳健。

## 安装

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

STEP 解析由 [cascadio](https://github.com/mikedh/cascadio)（OpenCASCADE 内核）完成，pip 直接可装，无需 conda。

## 快速开始

```powershell
# 1. 列出 STEP 文件中的实体
python lattice_fill.py model.step --list

# 2. 对 0 号实体做 Gyroid 填充: 单元 6mm, 壁厚 0.8mm, 表面皮壳 1.2mm
python lattice_fill.py model.step -s 0 -t gyroid -c 6 -w 0.8 --shell 1.2 -o out.stl

# 3. 蜂窝填充, 输出切片预览图便于目检, 未选中实体原样并入
python lattice_fill.py model.step -s 0,2 -t hex -c 4 --preview --keep-others
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-s, --solids` | `all` | 选取实体：逗号分隔的序号或名称子串，如 `0,2` 或 `Body1` |
| `-t, --type` | `gyroid` | 晶格类型，见下表 |
| `-c, --cell` | `5.0` | 晶格单元大小 (mm) |
| `-w, --wall` | `0.8` | 晶格壁厚 (mm)；hex/grid 为精确值，TPMS 为近似值 |
| `-d, --strut` | 取 `--wall` | 连杆直径 (mm)，用于杆状晶格 hex3d/bcc/cubic，精确值 |
| `--layer` | 取 `--cell` | 层高 (mm)，hexcell/hex3d 的 Z 向周期 |
| `--blur` | `0` | 隐式场高斯平滑 sigma（体素数，建议 0.8~1.2），消除表面体素台阶 |
| `--shell` | `1.0` | 保留的表面皮壳厚度 (mm)，外表面与内孔面均保留；`0` 不留皮壳 |
| `-r, --res` | `8` | 分辨率：每个晶格单元的体素数，越大越精细也越慢 |
| `--smooth` | `0` | Taubin 平滑迭代次数 |
| `--preview` | 关 | 每个实体额外输出 4 张 Z 向切片拼图 PNG |
| `--keep-others` | 关 | 未选中的实体以原始网格并入输出 |
| `--scale` | 自动 | 坐标缩放；STEP 自动按 m→mm 乘 1000 |
| `-o, --output` | `<输入>_lattice.stl` | 输出文件，后缀决定格式 (.stl/.obj/.ply/.glb) |

## 晶格类型

| 类型 | 结构 | 特点 |
|---|---|---|
| `gyroid` | Gyroid 极小曲面 (TPMS) | 各向近似均匀、自支撑性好，增材制造首选 |
| `schwarz` | Schwarz-P 曲面 (TPMS) | 孔道沿三轴正交贯通，利于排粉/排液 |
| `diamond` | Schwarz-D 曲面 (TPMS) | 刚度较高 |
| `iwp` | Schoen I-WP 曲面 (TPMS) | 壁面偏厚重的 TPMS 变体 |
| `hex` | 六边形蜂窝壁板（沿 Z 挤出） | 面内刚度高，竖直孔道贯通；面内封闭 |
| `grid` | 正交立方壁板 | 简单直观；**形成封闭立方腔**，粉末/树脂工艺慎用 |
| `hexcell` | **立体蜂窝壁板**：六边形棱柱壁 + 层间隔板 | 蜂窝腔封闭，结构致密（~40-60%），壁厚 `-w`、层高 `--layer` 可调 |
| `hex3d` | 立体蜂窝连杆框架：六边形顶点竖杆 + 每层六边形横杆框 | 全开放孔道，轻量（密度 ~15-25%），连杆直径 `-d` 精确可调 |
| `bcc` | 体心立方斜连杆（8 根体对角杆/单元） | 各向受力均衡，全开放，杆径精确可调 |
| `cubic` | 立方体边连杆（三轴向圆杆） | 最轻量，轴向刚度好，杆径精确可调 |

## 工作原理

1. **载入**：cascadio 把 STEP 转为三角网格场景，每个 STEP 实体独立列出，可按序号/名称选取。
2. **SDF**：对选中实体体素化（体素间距 = 单元大小 / `res`），做内外欧氏距离变换，
   得到毫米单位的符号距离场（内正外负）。
3. **皮壳**：`0 < sdf < shell` 的薄层保留为实心——外表面与内部孔洞面都会得到皮壳。
4. **晶格**：内部区域与周期隐式场求交。TPMS 用数值梯度把无量纲场归一化为近似真实
   距离，使 `--wall` 以毫米计；hex/grid 用解析距离，壁厚精确。
5. **提取**：皮壳与晶格取并集后 Marching Cubes 提取零等值面，清理退化面，输出水密网格。

## 精度与限制

- 输出是**网格近似**而非 B-rep：表面精度受体素间距限制（= `cell / res`）。
  需要更光滑的表面时增大 `--res` 或配合 `--smooth`。
- 保证特征不丢失的经验规则：体素间距 ≤ `wall/2` 且 ≤ `shell/2`（程序会警告）。
- STEP 默认按毫米建模处理；单位异常时用 `--scale` 修正（`--list` 会显示包围盒尺寸，可先核对）。
- 多个选中实体各自独立填充后合并输出；相互重叠的实体不做布尔并。
- 体素总数上限约 1.2 亿（内存保护），超限时程序会提示增大 `--cell` 或减小 `--res`。

## 验证示例

对 trimesh 官方测试零件 `featuretype.STEP`（127 × 63.5 × 34.9 mm，含通孔、沉孔、槽、缺口）：

```powershell
python lattice_fill.py test_data\featuretype.step -s 0 -t gyroid -c 8 -w 1.0 --shell 1.5 -r 10 --preview
```

切片预览可见：内部 Gyroid 连续填充，所有圆孔周围保留实心皮壳环，外轮廓与槽的表面完整保留；
输出水密，相对密度 ~44%。
