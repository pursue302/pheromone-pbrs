# Pheromone-inspired PBRS

强化学习课程项目：复现 2019 年蚁群启发 SARSA 奖励塑形，并研究实现修正、静态/动态/混合经验势函数及原始奖励上的实际效果。

## 快速开始

参考环境为 Python 3.13。进入本仓库目录后：

```bash
python -m pip install -r requirements.txt
python run.py list
python run.py verify
python run.py test
python run.py smoke
python run.py diagnostic
python run.py figures
```

`smoke` 实际运行 6 种方法、共 6000 步，只检查执行与数值；`figures` 从已保存快照重建 4 幅论文图和表格。

一个完整预算的新训练种子及其统计：

```bash
python run.py train validation --seeds 1 --scenarios u_detour --workers 2 --out demo
python run.py analyze fresh --input runs/demo --out demo_statistics
```

单种子仅验证流程。完整补充验证使用配置默认的 20 个独立种子：

```bash
python run.py train validation --workers 4 --out validation_repeat
python run.py analyze fresh --input runs/validation_repeat --out validation_statistics
```

## 内容

| 路径 | 用途 |
|---|---|
| `run.py` | 统一训练、检查、统计与绘图入口 |
| `configs/` | 11 组实验配置，明确方法、场景、种子和预算 |
| `workspace/paper_static_replication_20261120/` | 直接调用作者完整学习函数的近似复现、历史 Gym 兼容层、来源与补丁 |
| `workspace/research_story_20261120/` | 数值修正、静态/动态/混合、MC 与强基线 |
| `workspace/paper_report_20261120/experiments/` | 最新独立验证、状态势对照和目标保持诊断 |
| `tools/` | 独立输出的统计适配器 |
| `paper/` | 最新报告 PDF、LaTeX、Python 绘图和数据快照 |
| `docs/CODE_MAP.txt` | 方法与代码导航 |
| `docs/SOURCES.txt` | 论文、代码仓库、固定提交与归属 |

## 数据范围

研究对应 920 条作者近似复现、3660 条受控研究、600 条补充验证，共 5180 条学生训练记录及 4 个教师运行。
本 Git 仓库提供源码、配置、统计汇总、论文数据快照和图表。完整逐次记录、模型和轨迹保存在另行整理的离线证据包 `pheromone_pbrs_project_20261120.zip` 中，没有加入 Git 历史。

`docs/INVENTORY.json` 和 `docs/OFFLINE_PACKAGE_GUIDE.txt` 描述完整离线证据包；本仓库实际交付文件以根目录 `MANIFEST.json` 为准。
默认 `analyze fresh/controlled/source` 针对完整离线记录；只有源码时可重新训练，并通过 `--input` 分析新运行。恢复离线记录后可重新计算原始队列。
所有新运行输出写入 `runs/` 或 `analysis_results/`，已有目录不会被覆盖。

## 结果与贡献边界

评价重点为环境原始回报、原始折扣回报和成功率，塑形奖励不作为最终成绩。
独立验证支持 U 形绕行和四房间中相对原始信息素的部分改善；随机滑移优势没有复现。
简单状态势可以达到相近效果，复杂混合与蚁群权重的独立收益仍未得到稳定支持。
作者资格迹重置的实现收益单独报告，不算作新塑形方法的性能贡献。
完整结果、条件和局限见 [报告](paper/report.pdf)。

## 来源

- Bromuri (2019), [论文](https://doi.org/10.1007/s10732-019-09408-x)、[作者仓库](https://github.com/SB-BISS/RLACOSarsaLambda)。固定提交：`95786b39b9ee8c5e004e71ccfa790304cbb3f684`。
- 上游作者代码、历史 Gym 和 tilecoding 的许可证及来源保留在对应目录。
- 本仓库的实现、教师信息素与模型为本项目复现或生成，不是作者发布的实验产物。
- 代码和写作使用过 AI 辅助，实验依据为可执行代码、保存记录与冻结协议。

原学习与统计公式保持冻结；GitHub 版只调整 `run.py` 的清单校验和缺失离线数据时的提示，改动见 `docs/GITHUB_ADAPTER.diff`。
