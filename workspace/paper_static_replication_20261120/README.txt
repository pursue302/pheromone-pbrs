论文表7静态信息素效果复现尝试

正式训练直接运行paper_author中的作者完整learn()，不调用旧项目重写的TD循环。
旧项目experiment.py仅提供已经对原tile编码核对的索引投影和FastCoder；
本轮没有混合、探索奖励或自适应比例。

先看：复现结果.html、复现结果.txt、summary.json（正式测试完成后生成）。
PROTOCOL.txt是新结果出现前固定的设置；results/manifest.json保留源文件哈希。
run_replication.py是实验入口，analyze_replication.py使用作者原统计函数分析。
author_metrics.py直接提取自公开PlotResultsM3D.py的find_best_results，
包括从0计数的最佳回合，以及累计时不包含最佳回合本身的ATBS定义。

paper_author/：作者原SARSA、HA、PA完整类及最小修改副本。
PA_frozen.patch：只给局部更新补dynamic判断，实现真正静态；原开关另列。
PA_*eq38.patch：仅将源码TrueOnline塑形表达式改成论文式38，另作敏感性测试。
没有加入之前的资格迹清空、初始/终止势或成功标记改动。
gym_legacy/：历史Gymv0.10.5原MountainCar及seeding，只有import兼容。
upstream/：检索到的作者绘图/统计代码、历史Gym原文件及来源记录。
source_versions.json记录代码版本，build_sources.py保留生成过程。

results/priors：每位老师的200回合与成熟tau、原tile编码、实际成本和稳定记录。
  .pkl均是本轮自行生成的文件，仅迁移tau/coder；不冒充作者缺失的原文件。
results/teachers：4位老师完整训练回合、时间、质量和模型。
results/runs：920次学生训练，每次200完整回合、原始奖励和原作者指标。
results/models：对应最终Q、资格迹、信息素；初始Q均为0。
results/trajectories：主实验第一个种子的7种方法逐步交互样本。
smoke/为短小接入测试，不进入正式统计。

主对照：100个新学生种子；历史Gym、原脚本探索衰减、1000回合以上固定先验。
同一份信息素文件及编码供全部100次静态训练使用，贴近原脚本加载方式。
另保留早期先验、原静态开关、同编码无信息素控制，及另外两个老师的文件。
论文文字环境和论文式38与源代码不同，保留所有独立20种子歧义对照。
作者实际使用的Gym版本和原静态文件未知，不能声称已经恢复完全相同条件。

环境：Python3.13.5、NumPy2.1.3、SciPy1.15.3、Numba0.61.0、six1.17.0、
Matplotlib3.10.0。历史Gym源码直接移植，不安装旧Gym/Python2。
依赖相邻pheromone_feedback_2019/experiment.py和author_port（独立包会附带）。
生成与源码一致测试还读取author_direct_comparison_20261120/author_agents中的
5份原兼容源码及LICENSE，独立包同样附带，不需要旧实验的大量训练数据。

检查：python -m unittest -v test_replication
训练：python run_replication.py --workers 8
分析：python analyze_replication.py
训练拒绝覆盖已有manifest。复跑：python run_replication.py --out results_again --workers 8，
再python analyze_replication.py --out results_again（结果汇总保存在新数据的report中）。
不要在正式训练后重建或修改冻结的训练文件。

来源：https://github.com/SB-BISS/RLACOSarsaLambda
作者固定提交：95786b39b9ee8c5e004e71ccfa790304cbb3f684
论文：https://link.springer.com/content/pdf/10.1007/s10732-019-09408-x.pdf
历史环境：https://github.com/openai/gym/blob/v0.10.5/gym/envs/classic_control/mountain_car.py
