# 从零到前沿：机器学习系统自学路线（数学基础 → LLM / 扩散 / 推理模型）

> 一份可直接使用的中文自学课程文档。覆盖从高中数学到 2024–2026 前沿（推理模型、MoE、长上下文、多模态、Agent）的完整路线，强调"先建直觉再补严谨、能从零实现才算理解、数学服务于模型、读一手资料、项目驱动闭环"。
>
> **本版已吸收课程评审意见**：补齐了评测方法论、强化学习基础、数据工程、AI 安全与负责任 AI、可解释性、GNN/推荐系统概览、经典 NLP 铺垫、MLOps、Agent 系统设计等缺口；修正了 CLIP/RL/分词/RoPE 的先后顺序；消除了反向传播、分词、归一化、KL、生成模型对比等重复；并把"模型版图"等强时效内容改造为"可自我更新的判断框架"。

---

## 第一部分 · 课程总览

### 1.1 设计理念（七条核心哲学）

| 理念 | 含义 |
|---|---|
| **先建直觉再补严谨** | 每个概念先用几何/代码直觉跑通，再回头补形式化推导。拒绝一上来啃定理证明而失去动力，也拒绝只调 API 而不懂原理。 |
| **能从零实现才算理解**（implement-to-understand） | 核心机制（反向传播、注意力、扩散采样、LoRA、GRPO）都要求先用 NumPy/PyTorch 手写最小版本，再用框架。判据是"能复现"而非"能复述"。 |
| **数学服务于模型** | 线性代数/概率/最优化只学到能支撑后续模型的深度，遇到新需求再回炉加深。拒绝"先学完三本数学书再开始"的拖延陷阱。 |
| **沿信息流主线推进** | 数据 → 表示 → 架构 → 训练目标 → 优化 → 对齐 → 应用。Transformer（模块9）是全程枢纽。 |
| **读一手资料** | 经典论文、原始课程（CS231n/CS224n/CS336）优先于二手博客。建立"读论文—复现—质疑"的研究品味。 |
| **项目驱动、里程碑闭环** | 每阶段结束都有可展示的 capstone，把零散知识缝合成作品集。 |
| **拥抱"足够好"与迭代** | 前沿变化极快，目标不是学完所有东西，而是建立能持续自学新论文的底层能力与工程肌肉。 |

### 1.2 总时长

- **系统深入路线**：约 20–22 个月
- **工程应用快速路线**：约 10–12 个月
- **研究方向路线**：系统路线 + 论文复现，22+ 个月持续滚动
- 假设投入：**每周 10–15 小时**（理论输入 40% + 动手编码 40% + 复盘笔记 20%）

### 1.3 学习方法论（"直觉→数学→代码→复盘"四步法）

1. **直觉**：先看可视化/动画（3Blue1Brown、distill.pub、bbycroft.net/llm）。
2. **数学**：补形式化推导。
3. **代码**：亲手实现最小可运行版（Karpathy 的 micrograd → makemore → nanoGPT → minbpe 是脊柱）。
4. **复盘**：用费曼学习法讲给"假想的人"听，写进第二大脑（Notion/Obsidian/Anki，做间隔重复）。

**关键习惯**：
- 用 **masteryChecks** 做关卡判据，不达标不前进——后面的模块会无情放大前面的债务。
- **算力管理**：从 CPU/小数据集玩具规模起步（MNIST、tiny-shakespeare、CIFAR-10），需要 GPU 时用 Colab/Kaggle 免费额度或按需租云（Lambda/RunPod）。先验证正确性，再放大。
- **从阶段 2 起读论文**：用"三遍法"（先摘要+图表抓骨架，再方法抓核心，最后细节能复现）。
- **作品集导向 / learn in public**：每个 capstone 整理成 GitHub 仓库 + 简短博客。
- **善用 AI 助手做苏格拉底式提问，但不外包理解**：可问 LLM 解释/debug，但推导与实现必须自己跑通。警惕"看懂了"的错觉。

### 1.4 如何使用本课程

- **本课程把评审指出的横向能力做成了"贯穿支线"**：评测方法论、数据工程、AI 安全与伦理、可解释性，会在多个模块以小节嵌入并在 0.5/补充模块统一框架，避免散落。
- 每个模块都标注了**先修依赖**，并在三条路线中标注可略读节点。
- 螺旋上升的概念（如 KL 散度、注意力、缩放定律）在多处出现时，会明确是"复用/深化"而非"重讲"。

### 1.5 先修关系图（Prerequisite DAG，文字表达）

```
阶段0  M1(线代/微积分/矩阵求导) ─┐
       M2(概率/信息论/最优化/RL速成) ─┼─→ 全程地基
       M3(Python/NumPy/PyTorch/实验工程) ─┘
                    │
阶段1  M4(经典ML) ──→ M5(深度学习基础, 复用M1反传数学)
                    │
阶段2  M6(CNN/CV)   M7(RNN/序列/注意力雏形, +最小分词概念)
                    │            （M7 attention 是 M9 强先修）
阶段3  M8(表示学习/AE/VAE/GAN/嵌入/对比学习loss)
       M9(Transformer 枢纽 ★最高优先级, 分词深化, CLIP在此后)
                    │
阶段4  M10(扩散/多模态生成, 生成模型统一对比表)
       M11(LLM预训练/缩放/分布式/MoE, 数据工程一等公民)
                    │
阶段5  M12(对齐 SFT/RLHF/DPO, 依赖M2 RL速成)
       M13(PEFT/Prompt/RAG/Agent/推理优化)
       M14(前沿2024–2026: 推理模型/MoE/长上下文/多模态/Agent/版图)
                    │
贯穿   M-E(评测与方法论)  M-S(安全与负责任AI)  M-I(可解释性)
       概览补充(GNN/推荐/经典NLP/MLOps)
```

**关键先修断层修复（吸收评审）**：
- **RL 基础前置**：M2 末尾新增"强化学习速成"（MDP、策略梯度、价值函数、actor-critic、GAE、PPO 直觉），消除 M12 RLHF/PPO 的最严重先修断层。
- **CLIP 顺序修正**：M8 只讲对比学习损失（纯表示学习视角），**CLIP 的多模态对齐推迟到 M9 之后（并入 M10）**，因为 CLIP 文本编码器依赖 Transformer。
- **分词最小概念前置**：M7 前给出最小分词概念，M9 讲清算法，M11 只讲规模化工程取舍。
- **变分推断前置**：M2 信息论补 Jensen 不等式与变分下界直觉，支撑 M8 的 ELBO。

---

## 第二部分 · 学习路线（Tracks）

| 路线 | 时长 | 策略 | 适合 |
|---|---|---|---|
| **系统深入路线** | 20–22 月 | 阶段 0→5 全程顺序，每模块做 masteryChecks + 动手项目，核心机制坚持从零实现。重点投入阶段0（数学不偷工）与 M9（Transformer 打磨到逐行复现）。所有 capstone 全做。 | 立志做研究或追求彻底理解者 |
| **工程应用快速路线** | 10–12 月 | 数学只学到"能看懂、会用"（M1/2 抓重点、跳证明）；M4/6/7 取精华略读（CNN 与 RNN 任选其一精读，**但 M7 attention 不可跳**）；重点压在 M9 Transformer 直觉（理解即可）与阶段5（M12 概念级、M13 重度实操）。capstone 聚焦"微调+RAG+Agent"。M14 按需查阅。 | 尽快用 LLM 做产品者 |
| **研究方向路线** | 22+ 月持续 | 系统路线之上，从阶段2 起每周精读 1 篇论文，每阶段额外做 1 个复现实验。强化阶段4（扩散/LLM训练数学）与阶段5（对齐/推理模型/可解释性）。选定 1–2 个细分方向深耕。终点是能持续产出复现与改进。 | 立志做研究、形成研究品味者 |
| **数学/工程薄弱补强支线**（嵌入式） | 按需 | 任何路线中发现数学/工程拖后腿，回炉阶段0 对应薄弱主题（矩阵求导、概率、PyTorch 调试），遵循"数学服务于模型"，遇卡点再加深。 | 所有人 |

**三条路线的可略读标注约定**：模块内每个主题标注 `[全]`（全程必修）/`[研]`（研究路线深读）/`[快]`（快速路线可略读）。

---

## 第三部分 · 模块详情（按阶段组织）

---

# 阶段 0 · 数学与工具基础（10–14 周）

> 目标：把高中数学拉升到能读懂并推导 ML 论文的核心数学，并建立扎实的 Python 科学计算与深度学习实验工程能力。M1/2/3 可部分并行（上午数学、晚上敲代码，M3 贯穿始终）。

## 模块 1 · 线性代数与微积分（含矩阵求导与反向传播数学）

**一句话**：用"几何直觉 + 动手计算"双轨打通向量/矩阵/张量、特征值/SVD、梯度/雅可比/海森与矩阵求导链式法则，最终亲手推导反向传播——把它从"会调 `.backward()`"变成"知道每一步偏导从哪来"。

### 学习目标
- 用几何语言解释矩阵乘法（线性变换的复合）、行列式（面积/体积缩放与定向）、特征向量，并能在 NumPy 里画出二维线性变换前后的网格。
- 正确书写标量/向量/矩阵的各类导数，掌握分子布局/分母布局两套约定，知道为何深度学习里梯度 ∂L/∂W 与 W 同形。
- 从第一性原理推导常见矩阵求导公式（∂(Wx)/∂x、∂(xᵀAx)/∂x、∂‖x‖²/∂x 等），并用数值梯度验证。
- 把两层全连接网络写成计算图、手推每个参数梯度（含 softmax + 交叉熵的简洁梯度 ŷ−y）。
- **仅用 NumPy** 实现该网络的前向与反向，在 MNIST 子集上把损失训到下降。
- 用 SVD 解释 PCA、低秩近似、四个基本子空间与条件数；解释反向模式自动微分为何适合"多输入单输出"。

### 先修
高中代数与三角；一元微积分入门（导数定义、链式法则一元形式）；基础 Python/NumPy（与 M3 并行）；愿意"先直觉再严谨"的心态。

### 主题与要点（为什么重要）

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **向量/矩阵/张量与线性变换** `[全]` | 矩阵乘法=线性变换的复合（列=基向量变换后落点）；线性变换三性质；行列式=面积/体积缩放；张量=多维数组；形状契约 (m×n)·(n×p)=(m×p)；广播与转置陷阱 | 每一层=线性变换+非线性。理解几何后，前向传播张量形状、卷积、注意力 QKᵀ 都变"看得见" |
| **内积/范数/距离** `[全]` | aᵀb=‖a‖‖b‖cosθ；L1/L2/L∞/Frobenius；投影与正交分解；归一化与余弦相似度；标准正交基 | L2损失、权重衰减、余弦相似度、注意力分数、嵌入检索都建立在此 |
| **特征值/特征向量/对角化** `[全]` | Av=λv 几何画面；A=PDP⁻¹；对称矩阵谱定理；正定/半正定；谱半径与稳定性 | PCA、协方差、海森曲率、谱归一化、梯度消失/爆炸根因 |
| **SVD 与低秩结构** `[全]` | A=UΣVᵀ=旋转→缩放→旋转；奇异值=各方向增益；四个基本子空间；Eckart–Young 低秩近似；条件数=σ_max/σ_min；伪逆 | "线性代数的瑞士军刀"，直接是 LoRA 低秩微调的数学根基 |
| **导数/偏导/梯度/方向导数** `[全]` | 梯度指向上升最快方向、模长=最大方向导数；梯度与等高线正交；局部线性化 f(x+Δ)≈f+∇f·Δ；鞍点 | 梯度下降是训练一切的引擎 |
| **雅可比/海森与二阶信息** `[全]` | 雅可比 J=多元链式法则积木；det(J)=局部体积缩放；海森特征值=曲率；二阶泰勒展开；向量-雅可比积 VJP | 反向传播=一连串 VJP；海森解释优化难易 |
| **多元链式法则与计算图** `[全]` | 路径求和；计算图节点知道局部导数；正向 vs 反向模式；扇出梯度累加；detach | 通往"反向传播为什么成立"的总枢纽 |
| **矩阵求导：布局与核心公式** `[全]` | 四类导数；分子/分母布局；必背公式；形状检查法；微分法 dL=tr(Gᵀ dX) | 全连接层、注意力、归一化的梯度推导全靠这套 |
| **落到反向传播：公式到 NumPy** `[全]` | ∂L/∂z2=ŷ−y 起手回推；权重梯度=上游梯度⊗输入；batch 求和约定；梯度检查（相对误差<1e-7）；log-sum-exp | 阶段0 capstone 核心，亲手推+写才真正拥有反向传播 |

> **吸收评审（消除重复）**：反向传播在 M1/M3/M5 三处分工明确——**M1 只负责数学推导**（公式 + 一个最小 NumPy 验证），完整工程实现留给 M5。

### 关键资源
- **3Blue1Brown《线性代数的本质》/《微积分的本质》/《神经网络》** (video) — 几何直觉打底首选。
- **Gilbert Strang《Introduction to Linear Algebra》+ MIT 18.06** (course) — 四个基本子空间、SVD 讲得最透。
- **The Matrix Cookbook** (book) — 矩阵求导速查工具书。
- **《Mathematics for Machine Learning》(Deisenroth et al.)** (book, 免费) — 专为 ML 写的数学主线。
- **CS231n 反向传播/向量矩阵梯度讲义** (course) — 手写 BP 前必读。
- **Karpathy《building micrograd》(YouTube + repo)** (video) — implement-to-understand 范本。
- **Parr & Howard《The Matrix Calculus You Need For Deep Learning》(explained.ai)** (blog) — 矩阵求导最佳单篇。

### 动手项目
1. 可视化线性变换（旋转/缩放/剪切 + 特征向量验证方向不变）。
2. 手算+数值梯度验证 5 个矩阵求导公式（相对误差<1e-6，产出对照表）。
3. SVD 图像压缩（k=5/20/50 重建对比 + 奇异值衰减曲线）。
4. **纯 NumPy 两层全连接网络 + 反向传播**（线性→ReLU→线性→softmax→CE，加梯度检查，MNIST 子集）。
5. 跟做 micrograd（标量自动求导引擎 + 玩具 MLP）。
6. 梯度几何小实验（病态二次型等高线 + 梯度场 + 之字形轨迹 + 海森特征值）。

### 常见误区
- 把矩阵乘法当"机械行乘列"死记，从不建立几何图像。
- 混用分子/分母布局导致该转置的没转置 → 固定一套约定 + 永远形状检查。
- 推完公式不做数值梯度验证就写进代码。
- 跳过手写反向传播直接上 `.backward()`，遇 NaN/梯度消失无从下手。
- softmax/交叉熵不做 log-sum-exp 稳定化导致 NaN。
- 陷入"先读完三本数学书"的拖延；特征值/SVD 深水区可在 PCA/LoRA 用到时回炉。
- 混淆特征分解（需方阵）与 SVD（任意矩阵）。
- 忽视 batch 维求和/平均约定。
- 只看视频获得"懂了的错觉"，从不动笔。

### 时间估计
约 3–4 周（35–55h）。第1周线代直觉+可视化；第2周内积/范数/特征值/SVD+压缩；第3周导数/梯度/雅可比/海森/链式法则；第4周矩阵求导+手写两层网络+micrograd。

### 掌握自检
能画二维剪切矩阵对网格的作用并说清行列式/特征向量；能推 ∂(xᵀAx)/∂x=(A+Aᵀ)x；能说清非方阵 SVD 中 U/Σ/V 形状与几何；能解释"梯度指向上升最快"与"梯度与等高线正交"为何同一件事；能把两层网络画成计算图手推 ∂L/∂W；能纯 NumPy 实现并通过梯度检查；能解释反向模式为何高效、本质是 VJP 链；能写 log-sum-exp 稳定化；能用海森特征值判断临界点类型。

---

## 模块 2 · 概率统计、信息论与最优化（含强化学习速成）

**一句话**：从随机变量与贝叶斯出发，经 MLE/MAP 把"概率假设"翻译成"损失函数"，用信息论（熵/交叉熵/KL/互信息）解释损失为何长这样，再用凸性与梯度下降/SGD/Adam 把损失优化下去——三条线在"最小化损失"汇合，并为 VAE 的 ELBO 与 RLHF 的 KL 约束埋伏笔。**末尾新增"强化学习速成"消除 M12 先修断层。**

### 学习目标
- 说清随机变量/PMF/PDF/CDF；写出伯努利、二项、Categorical、高斯、均匀、指数、泊松分布及用途。
- 计算期望/方差/协方差/相关系数，理解期望线性性质与协方差矩阵几何。
- 运用条件概率、全概率、贝叶斯定理，分清先验/似然/后验/证据。
- 从"i.i.d.+概率模型"推导 MLE，说明 MLE ⟺ 最小化 NLL；加先验得 MAP，解释 MAP 对应 L2/L1 正则。
- 写出熵/交叉熵/KL/互信息定义，证明"最小化交叉熵 ⟺ 最小化 KL ⟺ 最大化似然"三者等价。
- 解释分类用交叉熵、回归用 MSE 各自追溯到 Categorical/高斯 MLE。
- 判断凸性，手写 GD/SGD/动量/Adam 更新公式；解释学习率、warmup、二阶方法思想。
- **（新增 RL 速成）** 说清 MDP、状态/动作/奖励/策略、价值函数、策略梯度、actor-critic、GAE 与 PPO 的直觉。
- 讲清 KL 将在 VAE 的 ELBO 与 RLHF 奖励约束中的角色（埋点）。

### 先修
M1 的梯度/链式法则/向量矩阵运算；会求一元极值与简单定积分；Python/NumPy 入门；对数/指数性质。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **随机变量与常见分布** `[全]` | PMF（离散）vs PDF（连续，值可>1）；伯努利/二项/Categorical/高斯/均匀/指数/泊松的物理意义；高斯一维+多维（协方差椭圆）；CLT；配分函数 | 模型输出层、损失、初始化、噪声假设全用分布描述 |
| **期望/方差/协方差** `[全]` | 期望线性性质（即使不独立）；方差/协方差矩阵（对称半正定，呼应 M1 特征分解）；无偏估计；蒙特卡洛估计 | 每个损失本质是"某个量的期望"；SGD 梯度是真梯度的无偏估计 |
| **条件概率与贝叶斯** `[全]` | P(A\|B)；链式法则（自回归 LM 的数学根 P(x)=∏P(xₜ\|x_<t)）；posterior∝likelihood×prior；基率谬误；条件独立；贝叶斯 vs 频率派 | 判别/生成区别、MLE/MAP、朴素贝叶斯、VAE 后验、RLHF 策略更新都在此 |
| **MLE 与 MAP** `[全]` | 对数似然把连乘变连加；高斯MLE→MSE、伯努利/Categorical MLE→交叉熵；MAP：高斯先验→L2、拉普拉斯→L1；为何用对数；小样本过拟合 | 枢纽：把"数据怎么生成"翻译成"该最小化什么" |
| **采样基础** `[全]` | 逆变换采样；**重参数化预告** μ+σ·ε（VAE 能反传的关键）；拒绝/重要性采样、MCMC 概念；随机种子可复现 | VAE/扩散/RL 策略采样/LLM 解码全是采样 |
| **熵与交叉熵** `[全]` | 自信息 −log p；熵=平均不确定性；交叉熵 H(p,q)≥H(p)；分类交叉熵=one-hot 与 softmax 间交叉熵=NLL；log-sum-exp；CrossEntropyLoss 内含 log-softmax | 深度学习最常用损失，从图像分类到 LM 预训练 |
| **KL 散度与互信息** `[全]` | D_KL(p‖q)=H(p,q)−H(p)；非负、不对称、非距离；最小化CE⟺最小化KL（三线汇合）；前向 vs 反向 KL（矩匹配 vs 寻峰）；互信息=表示学习理论基础；**埋点：VAE 的 ELBO、RLHF 的 KL 惩罚** | 连接信息论与现代生成/对齐模型的总枢纽 |
| **凸性与优化几何** `[全]` | 凸集/凸函数/Hessian 半正定；深网络非凸但可训（敌人是鞍点/病态/坏初始化）；负梯度=最速下降；条件数与病态 | 决定损失能否真正降下去 |
| **GD/SGD/动量/Adam** `[全]` | full-batch vs SGD vs mini-batch；动量=梯度指数滑动平均；AdaGrad→RMSProp→Adam（动量+自适应+偏差修正）；**AdamW**（解耦权重衰减，Transformer 标准）；手写对比 | 把"损失"变"训练好的模型"的引擎 |
| **学习率/调度/二阶概览** `[全]` | 学习率最重要；step/cosine + 线性 warmup；梯度裁剪；牛顿法 O(d³) 不可行 → 几乎都用一阶；Adam=廉价近似二阶 | warmup+cosine 是现代大模型标配 |
| **★ 强化学习速成（新增，吸收评审）** `[全]` | MDP（状态/动作/奖励/转移/折扣）；策略 π、回报 G、价值函数 V/Q；策略梯度 ∇J=E[∇log π·A]；REINFORCE 的高方差与基线；actor-critic；优势 A=Q−V 与 **GAE**；**PPO 直觉**（裁剪代理目标、信任域、KL 约束）；on/off-policy | **消除 M12 RLHF/PPO 与 M14 RL-for-reasoning 的最严重先修断层** |

> **吸收评审**：RL 速成放在 M2 末尾（也可在 M11 与 M12 之间复习），让学员到 PPO/GRPO 时不再遇到未学概念。M2 信息论补 **Jensen 不等式与变分下界直觉**，前置 M8 的 ELBO 推导工具。

### 关键资源
- **3Blue1Brown — Probability / CLT / Bayes** (video) — 几何直觉首选。
- **StatQuest (Josh Starmer)** (video) — 概念"蠢萌好懂"，按主题点播。
- **MIT 6.041 / 6.431x (Tsitsiklis)** (course) — 权威概率第一课。
- **Deep Learning（花书）第3/4/5/8章** (book, 免费) — 概率+信息论+优化精炼综述。
- **PRML (Bishop) 第1.6/第2章** (book) — 贝叶斯视角黄金标准。
- **Sebastian Ruder《gradient descent overview》** (blog) — 优化器最高效单点资源。
- **Adam (2015) / AdamW (2019) 原论文** (paper)。
- **Convex Optimization (Boyd) + EE364A** (book/course, 免费)。
- **Distill《Why Momentum Really Works》** (blog) — 交互可视化。
- **（RL 速成）Sutton & Barto《Reinforcement Learning》第3/13章 + OpenAI Spinning Up（PPO/GAE）+ Hugging Face Deep RL Course** — RL 入门与 PPO 落地。
- **Karpathy makemore / Zero to Hero** (repo) — 把概念落到代码。

### 动手项目
1. 分布动物园（采样直方图 vs 理论曲线 + 二维高斯协方差椭圆，联系 M1 特征向量）。
2. 贝叶斯推断（医疗检测后验曲线扫基率 + 手写朴素贝叶斯垃圾邮件 vs sklearn）。
3. 从概率假设到损失（高斯回归 MLE→最小二乘、Categorical→交叉熵、MAP 先验强度扫描）。
4. 信息论三件套验证（entropy/cross_entropy/kl/mutual_information，验证 H(p,q)=H(p)+D_KL）。
5. 优化器擂台（病态二次/Rosenbrock 上 GD/SGD/动量/Adam 轨迹 + cosine 调度对比）。
6. 微型 softmax 分类器（缝合"分布→MLE→交叉熵→SGD+动量→L2"整条链路）。
7. **（RL 速成）** 在 CartPole 或 GridWorld 上跑通 REINFORCE 与一个最小 actor-critic，观察基线如何降方差。

### 常见误区
- 把 PDF 值当概率（连续分布只有积分才是概率）。
- 期望线性性质只对线性运算成立：E[XY]≠E[X]E[Y]、E[g(X)]≠g(E[X])（Jensen 来源）。
- 死记交叉熵不知它=Categorical NLL。
- 把 KL 当距离（不对称、无三角不等式）。
- 数值不稳定（小概率取 log、连乘下溢、softmax 不减最大值）。
- 对 CrossEntropyLoss 再手动 softmax。
- 因"非凸"恐惧深度学习（真正的敌人是鞍点/病态/坏初始化/学习率）。
- 把 Adam 当万能默认。
- 忽视 warmup/调度致 Transformer 初期 NaN。
- MLE 小样本必过拟合（拉普拉斯平滑/MAP 来救）。
- **（RL）** 把策略梯度的高方差误判为 bug；忘了基线/优势能大幅降方差。

### 时间估计
约 3.5–4.5 周（含 RL 速成约 +0.5–1 周）。第1周分布/期望方差协方差；第2周贝叶斯/MLE/MAP/采样；第3周信息论；第4周最优化；第4.5–5周 RL 速成。

### 掌握自检
能从 i.i.d. 推 MLE 并说明 MLE⟺最小化 NLL，分别推出高斯→MSE、Categorical→交叉熵；能证 H(p,q)=H(p)+D_KL 并说清"优化交叉熵"而非"KL"的原因；能做贝叶斯数值题并解释基率谬误；能说清高斯先验↔L2、拉普拉斯↔L1；能纯 NumPy 实现 GD/SGD/动量/Adam 并解释偏差修正；能完整训练 softmax 分类器并逐项对应概念；能处理 softmax 数值稳定；能预告 KL 在 ELBO/RLHF 的角色；**（RL）能写出策略梯度公式、解释优势函数与 GAE、说清 PPO 裁剪目标在做什么。**

---

## 模块 3 · Python、NumPy、PyTorch 与实验工程

**一句话**：从科学计算栈到能从零写出完整、可复现训练循环的深度学习工程地基——吃透 NumPy 向量化/广播、PyTorch 张量/autograd/nn.Module/训练循环、数据加载、GPU/混合精度，并建立实验管理、调参调试与 Hugging Face 生态的工程肌肉。

### 学习目标
- 把含 for 循环的数值代码改写成纯向量化，并用广播规则解释 shape 变化与内存开销。
- 从零写不依赖高层封装的训练循环（zero_grad→forward→loss→backward→step），解释每行作用与漏写后果。
- 解释 autograd 计算图与反向传播，正确用 requires_grad / detach / no_grad / retain_graph。
- 用 nn.Module/DataLoader/调度器/混合精度组装工业级可复现训练脚本。
- 接入 W&B 或 TensorBoard，落地可复现性清单。
- 用 Hugging Face datasets/transformers 跑通推理与最小微调。
- 系统化调试训练故障（NaN/不降/OOM/CPU 瓶颈/train-val 不一致）。

### 先修
基础 Python；M1 线代（看懂张量 shape）；M1/M2 链式法则、梯度下降、交叉熵；命令行；可用 Python 环境（Miniconda 或 Colab）。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **Python 科学计算栈与环境工程** `[全]` | conda/venv 隔离环境 + 版本锁定；NumPy/SciPy/Pandas/Matplotlib/sklearn 分工；Jupyter/Colab 隐藏状态陷阱；何时从 notebook 毕业到 .py + argparse；pdb/breakpoint/assert | 多数"跑不出来"是环境/版本/隐藏状态而非算法 |
| **NumPy 数组/向量化/广播** `[全]` | shape/dtype/view vs copy；向量化快 10–100×；广播尾部对齐规则；花式索引/布尔掩码/axis/keepdims；数值稳定（logaddexp、softmax 减最大值） | 张量心智模型 1:1 迁移到 PyTorch |
| **PyTorch 张量与设备/dtype** `[全]` | Tensor vs ndarray（device/dtype/requires_grad）；from_numpy 零拷贝陷阱；view 要连续 vs reshape；`.to(device)`；float32/bf16/int64；in-place 风险 | 设备/dtype 不匹配是最高频报错 |
| **Autograd 自动微分** `[全]` | 动态计算图；requires_grad/leaf/grad_fn；**.grad 累加 → 必须 zero_grad**；no_grad/detach/freeze 三场景；retain_graph/create_graph | 区别于 NumPy 的核心魔法，但不是黑箱（呼应 M1，**此处只讲机制不重推数学**） |
| **nn.Module 与模型构建** `[全]` | __init__ 注册子模块/forward 定义计算；nn.Parameter/ModuleList（普通 list 不注册=经典坑）；train/eval 模式；nn.init；state_dict 保存 | 把张量运算组织成可复用/保存/迁移的模型 |
| **★ 从零写训练循环（核心）** `[全]` | 五步骨架默写；CrossEntropyLoss 喂 logits；epoch/batch + 验证分离 + no_grad；调度器何时 step；梯度裁剪/累积 | 阶段0 masteryCheck 硬指标，后续所有模型共用 |
| **数据加载 Dataset/DataLoader** `[全]` | __len__/__getitem__；batch_size/shuffle/num_workers/pin_memory；**collate_fn 处理变长 padding**（NLP 必备）；transforms 只训练加增强；数据管线是隐形瓶颈 | 模型再快数据喂不上也白搭 |
| **GPU/CUDA 与混合精度** `[全]/[快略]` | GPU 并行；避免循环里 `.item()/.cpu()` 同步；显存构成（参数+激活+优化器+梯度）；**AMP autocast+GradScaler + loss scaling**；bf16 vs fp16；torch.compile；DDP 概念 | LoRA/QLoRA 的前提；混合精度省一半显存提速 |
| **实验管理与可复现性** `[全]` | W&B/TensorBoard 记录 loss/指标/超参/资源；可复现清单（种子+cudnn+环境锁定+数据划分+commit hash）；严格可复现 vs 结果可比；一次只改一个变量 | 没有追踪就无法判断哪组最好 |
| **调参与调试方法论** `[全]` | 故障诊断树（NaN/不降/OOM/过拟合）；**先过拟合一个 batch**；shape/device/dtype 排查；学习率最重要（lr range test）；网格/随机/贝叶斯搜索 | 真实训练 90% 时间在调试 |
| **Hugging Face 生态入门** `[全]/[快深]` | AutoModel/AutoTokenizer/pipeline；datasets `.map()`；**Trainer 内部就是你手写的循环**；Hub/accelerate | 通往阶段5 的工具桥梁，先会手写再用封装 |

> **吸收评审（时效性）**：混合精度的数值原理（fp16/bf16 动态范围、loss scaling）在 M3 作**概念引入**，真正规模化落地在 M11；避免过早深挖。

### 关键资源
- **PyTorch 官方教程（Learn the Basics / 60 Minute Blitz）** (course)。
- **《Deep Learning with PyTorch》(Manning, 免费 PDF)** (book) — 核心开发者参与。
- **Karpathy《building micrograd》** (video/repo) — 理解 autograd 最佳一课。
- **NumPy 官方 Broadcasting 文档** (blog)。
- **CS231n Python/NumPy + PyTorch tutorial** (course)。
- **W&B Quickstart + PyTorch TensorBoard 教程** (blog)。
- **Hugging Face NLP Course** (course)。
- **Karpathy《A Recipe for Training Neural Networks》** (blog) — 训练调试圣经。
- **PyTorch AMP recipe + torch.compile 教程** (blog)。

### 动手项目
1. NumPy 向量化重写挑战（欧氏距离矩阵/softmax/一维卷积，`%timeit` 对比 + shape 推导）。
2. 纯 NumPy 两层网络（直通阶段0 capstone，参照 micrograd 验证梯度）。
3. **从零 PyTorch 训练循环**（不用高层封装，MNIST/FashionMNIST MLP，能默写每行）。
4. autograd 透视实验（打印 grad/grad_fn，分别试 detach/no_grad/漏写 zero_grad）。
5. 可复现 + 实验追踪脚本（接入 W&B，跑两次验证可比，对比有无 BatchNorm）。
6. 混合精度与 OOM 实战（fp32/amp/amp+累积 三配置显存吞吐对比表）。
7. 调试方法论演练（植入 5 个经典 bug 逐一定位）。
8. Hugging Face 跑通（pipeline 推理 + Trainer 微调 distilbert，与手写循环逐项对照）。

### 常见误区
张量与模型不在同一设备；忘记 zero_grad/backward/step；验证忘 eval()/no_grad()；对 CrossEntropyLoss 手动 softmax；普通 list 装子模块；循环里频繁 `.item()/.cpu()` 同步 + 累加张量拖着计算图致显存泄漏；view 对 transpose 后张量报错（需 contiguous）；切片返回 view 就地修改改到原数组；广播隐形 bug (N,1)−(N,)→(N,N)；notebook 乱序执行隐藏状态；不固定种子就调参；NaN 盲调；跳过"过拟合一个 batch"；直接用 Trainer 跳过手写循环。

### 时间估计
约 4–6 周（50–80h），与 M1/2 并行。"从零写训练循环"与"NumPy 两层网络"打磨到能默写；HF 跑通即可。

### 掌握自检
能默写完整训练脚本并逐行解释；能把 for 循环改写成向量化并推导 shape；能解释 zero_grad/backward/no_grad/detach/eval 及漏用后果；能手画计算图说明 backward 与 .grad 累加；面对 NaN/不降/OOM 能给系统排查顺序；能接入 W&B 落地可复现；能写 AMP 代码并解释 loss scaling/bf16 vs fp16；能用 HF 加载推理+Trainer 微调并指出对应手写循环哪部分。

---

# 阶段 1 · 经典机器学习与深度学习基础（10–13 周）

> 目标：从"最小化损失"统一视角理解监督/无监督学习，掌握偏差-方差/正则化/泛化；进入神经网络吃透反向传播、优化器、初始化、正则化与训练技巧。建立"模型=架构+损失+优化+数据"的统一心智模型。M4 是 M5 的概念铺垫。

## 模块 4 · 经典机器学习

**一句话**：用"模型=架构+损失+优化+数据"的统一视角吃透监督/无监督经典算法、模型评估与特征工程，建立贯穿全程的偏差-方差、正则化、泛化心智模型，并理解经典 ML 在工业界为何不可替代。

### 学习目标
- 用统一"损失+优化"视角解释线性/逻辑回归、SVM、决策树、k-means 的本质区别（假设空间/损失/优化/正则化四维）。
- 从零实现线性回归（正规方程+梯度下降）、逻辑回归（sigmoid+交叉熵+手推梯度）、决策树、k-means，对比 sklearn。
- 手推 L2 解析解、从约束几何/次梯度解释 L1 稀疏；实验展示 λ 移动偏差-方差。
- 讲清 SVM 最大间隔、对偶/支持向量、核技巧；bagging 降方差 vs boosting 降偏差，调参 XGBoost/LightGBM。
- 为分类/回归选对指标，设计无泄漏交叉验证，用学习曲线诊断过拟合/欠拟合并对症下药。
- 完成端到端特征工程流水线。

### 先修
M1 线代+微积分；M2 概率/信息论/最优化（含拉格朗日乘子）；M3 Python/NumPy/Pandas/Matplotlib。无需深度学习——本模块正是 M5 的概念地基。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **统一视角：损失最小化** `[全]` | 假设空间+损失+优化；经验风险 ERM vs 期望风险=泛化鸿沟；参数 vs 非参数模型 | 把十几个算法串成一张图的"元知识"，深度学习只是换模型/优化 |
| **线性回归** `[全]` | 正规方程闭式解 vs 梯度下降；高斯噪声 MLE=最小二乘；投影几何；多重共线性 | 麻雀虽小五脏俱全 |
| **逻辑回归与分类** `[全]` | sigmoid+交叉熵（伯努利 MLE）；为何不用 MSE；softmax 多分类；概率校准 | "神经网络的一个神经元"，工业主力 |
| **正则化 L1/L2** `[全]` | 岭回归解析解 w=(XᵀX+λI)⁻¹Xᵀy；L1 稀疏（菱形 vs 圆/次梯度）；Elastic Net；L2↔高斯先验、L1↔拉普拉斯（MAP）；λ=偏差-方差旋钮 | 对抗过拟合核心，贯穿全程（weight decay/dropout/标签平滑都是正则化） |
| **SVM 与核方法** `[全]/[快略对偶]` | 最大间隔；软间隔 C；hinge loss=SVM 纳入统一框架；对偶与支持向量；核技巧 K(x,x')；RBF/多项式核 | 深度学习前的统治算法，对偶/核训练对理解注意力有迁移价值 |
| **决策树与集成 RF/GBDT** `[全]` | 信息增益/基尼分裂；bagging（RF 降方差）vs boosting（GBDT 降偏差）；XGBoost/LightGBM（二阶近似、直方图、leaf-wise） | 表格数据"打不过"的模型，偏差-方差最生动实战 |
| **聚类 k-means/层次** `[全]` | k-means=EM 特例（分配-更新）；k-means++；选 k（肘部/轮廓）；层次聚类树状图；DBSCAN/GMM 拓展 | k-means 为 VAE/GMM 埋伏笔 |
| **降维 PCA/t-SNE/UMAP** `[全]` | PCA=协方差特征分解=SVD=最大方差=最小重构误差三等价；t-SNE 只是可视化（不保距/全局）；UMAP | PCA=表示学习的线性前身，直通 M8 自编码器 |
| **模型评估 I：划分与交叉验证** `[全]` | train/val/test 职责分离；k 折/分层/时序切分；**数据泄漏**（预处理必须折内 fit，用 Pipeline）；嵌套 CV | 评估错则结论全错，是所有 ML（含 LLM 评测）可信度基石 |
| **模型评估 II：偏差-方差诊断** `[全]` | 泛化误差=偏差²+方差+噪声；学习曲线/验证曲线；对症清单；双下降对照 | 从"瞎试"到"有依据决策"的关键能力 |
| **模型评估 III：指标** `[全]` | 混淆矩阵/precision/recall/F1；不平衡下准确率误导；ROC/AUC（概率解释）vs PR；RMSE/MAE/R² | 选错指标=优化错目标 |
| **特征工程** `[全]` | 缺失值/异常值；标准化（线性/SVM/kNN/k-means/PCA 必需，树不需）；类别编码（目标编码须折内防泄漏）；特征构造/选择；Pipeline/ColumnTransformer | "特征决定上限"，工业表格高度依赖 |
| **何时不用深度学习** `[全]` | 表格数据 GBDT 常胜；可解释/合规（评分卡、SHAP）；成本/延迟/小数据；先建强基线再判断升级 | 纠正"万物皆深度学习"偏见 |

> **吸收评审（缺口补充）**：本模块在 M4 概览层面补充 **GNN/推荐系统/表格深度学习** 的存在与归纳偏置（见"概览补充"小节链接），符合"归纳偏置随数据结构变化"的核心哲学；**可解释性**（SHAP/LIME/部分依赖）在此首次系统出现，呼应贯穿支线 M-I。

### 关键资源
- **ISL/ISLP (James et al., 2023)** (book, 免费) — 经典 ML 最佳入门主线。
- **ESL (Hastie et al.)** (book, 免费) — 进阶查阅手册。
- **Andrew Ng ML Specialization / CS229** (course) — 直觉+严谨。
- **scikit-learn User Guide** (blog) — 世界级工程文档。
- **《Hands-On ML》(Géron) 第1部分** (book) — 工程实战主线。
- **XGBoost (2016) + LightGBM (2017) 论文** (paper)。
- **t-SNE 论文 + distill《How to Use t-SNE Effectively》** (paper/blog)。
- **StatQuest** (video)。
- **Kaggle Titanic/House Prices + Learn 微课** (course)。
- **《Interpretable ML》(Molnar)** (book, 免费) — SHAP/LIME，呼应可解释性支线。

### 动手项目
1. 从零线性回归（正规方程+GD+岭回归，对齐 sklearn）。
2. 从零逻辑回归（sigmoid+交叉熵+手推梯度+决策边界）。
3. 从零决策树+k-means（基尼/信息增益/k-means++/肘部/轮廓）。
4. L1 vs L2 正则化路径图（展示 L1 压到 0）。
5. SVM 核方法可视化（make_moons/circles，线性 vs RBF，扫 C/γ）。
6. 梯度提升表格实战（完整 Pipeline + 交叉验证调 XGBoost/LightGBM + 特征重要性 + Kaggle 提交）。
7. PCA+t-SNE/UMAP 可视化（MNIST，扫 perplexity）。
8. **评估与诊断综合项目**（不平衡数据 + 故意制造泄漏 + 混淆矩阵/ROC/PR + 学习曲线诊断，阶段1 capstone 核心）。

### 常见误区
测试集污染；数据泄漏（预处理在划分前）；不平衡盲信准确率；忘记特征缩放（树不需）；过度解读 t-SNE 图；k-means 当万能聚类；GBDT 不调学习率/树数配合+不早停；只调模型不做特征工程；混淆 bagging/boosting 作用；正则化方向搞反；"复杂模型一定更好"；不设种子/不记录配置。

### 时间估计
5–7 周（60–90h）。工程快速路线可压到 4 周（略读 SVM 对偶，重 GBDT+评估+特征工程）。

### 掌握自检
能填四维表格；能从零实现线性/逻辑回归并对齐 sklearn；能手推岭回归解析解+解释 L1 稀疏；能讲清 SVM 最大间隔/软间隔/支持向量/RBF；能说清 RF vs GBDT 机制并调出优于基线的 GBDT；不平衡任务能选对指标+读 ROC/PR+解释 AUC；能设计零泄漏 Pipeline+用学习曲线诊断；能讲三个"该用经典 ML"场景。

---

## 模块 5 · 深度学习基础：从感知机到能训练的神经网络

**一句话**：吃透"模型=架构+损失+优化+数据"统一心智模型，亲手推导反向传播并用纯 NumPy 从零实现能在 MNIST 上收敛的多层网络，再系统掌握让深网络"训得动"的全套工程武器：激活函数、初始化、正则化、归一化、优化器与学习率调度。

### 学习目标
- 用几何与计算图两种视角解释感知机、MLP、通用逼近定理；说清单层为何解不了 XOR。
- 从标量链式法则手推两层网络（ReLU+softmax 交叉熵）完整反向传播，写出矩阵形式。
- **纯 NumPy** 实现前向/反向/SGD+动量/mini-batch，MNIST 上 >97%。
- 解释 sigmoid/tanh/ReLU/LeakyReLU/GELU 优缺点与死亡 ReLU；从方差守恒推 Xavier/He 初始化。
- 解释 Dropout/权重衰减/早停/数据增强机制差异，正确实现 inverted dropout。
- 推导 BatchNorm 前向/反向，解释 train/eval 与 running stats；说清 LayerNorm 为何适合序列。
- 用梯度检验验证实现，诊断梯度消失/爆炸；解释 SGD/Momentum/RMSProp/Adam/AdamW 差异，配置 warmup+cosine，产出训练技巧消融报告。

### 先修
M1 矩阵求导/链式法则；M2 交叉熵/MLE/梯度下降/期望方差；M3 NumPy 向量化/PyTorch/训练循环；M4 损失/过拟合/偏差-方差/正则化统一视角。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **感知机/MLP/通用逼近** `[全]` | 感知机=线性+阶跃，XOR 反例；非线性是关键（否则多层=单层）；通用逼近"存在≠可学到"；计算图统一语言；张量形状记账 | 从线性到深度的概念跳板 |
| **★ 反向传播：手推+梯度检验** `[全]` | 链式法则在计算图上高效复用；标量→矩阵形式（dW=xᵀ·d_out, db=Σd_out, dx=d_out·Wᵀ）；softmax+CE 联合梯度 (p−y)/B；局部×上游梯度路由；梯度检验<1e-7 | 整个深度学习的引擎（**此处做完整实现，呼应 M1 只推数学**） |
| **激活函数：ReLU 族** `[全]` | sigmoid/tanh 饱和→梯度消失；ReLU 不饱和但死亡；LeakyReLU/GELU/SiLU；激活与初始化/归一化耦合 | 决定梯度能否健康流过深层 |
| **损失与输出层** `[全]/[快略]` | MSE/Huber vs 交叉熵；softmax 数值稳定（减最大值/log-softmax）；为何分类不用 MSE；标签平滑/focal | 连接概率建模与优化的桥梁 |
| **初始化 Xavier/He** `[全]` | 全零破坏对称性失败；方差守恒推 Xavier（tanh）/He（ReLU）；初始化错误→信号消失/爆炸 | BatchNorm/残差前是生死线 |
| **正则化** `[全]` | inverted dropout（训练除 keep_prob）；weight decay=L2（Adam 需 AdamW）；早停；数据增强；先判过/欠拟合再加正则 | 控制泛化，对应 M4 偏差-方差 |
| **归一化 BatchNorm/LayerNorm** `[全]` | BN 对特征在 batch 维标准化+γβ；train/eval 与 running stats；BN 反向推导；**LayerNorm 沿特征维（RNN/Transformer 用）**；RMSNorm | 让深网络训到上百层的关键 |
| **优化实践** `[全]` | SGD→Momentum→RMSProp→Adam→AdamW；学习率最重要+lr range test+warmup+cosine；梯度裁剪；消失/爆炸成因（谱半径）与对策链 | 把"架构+损失"变可用模型的最后一公里 |

> **吸收评审（消除重复）**：归一化在 M5 讲清 BatchNorm 基础与 LayerNorm 动机；**M9 不再重复 BN 基础**，只差异化讲 Transformer 为何用 LayerNorm/RMSNorm。反向传播的完整 NumPy 实现集中在本模块；**阶段0 与阶段1 capstone 差异化**——阶段0=micrograd（标量自动求导引擎），阶段1=两层 MLP+MNIST（手写每层 backward + 模块化组件消融），避免重复。

### 关键资源
- **CS231n Module 1/2 笔记** (course) — 反向传播/激活/初始化/BN/优化最佳讲义。
- **花书第6/7/8章** (book, 免费) — 权威严谨参考。
- **Michael Nielsen《Neural Networks and DL》第2章** (book, 免费) — 反向传播四方程直觉。
- **Karpathy Zero to Hero（micrograd/makemore）** (video)。
- **karpathy/micrograd repo** (repo)。
- **BatchNorm/Dropout/Adam/He init 原论文** (paper)。
- **Karpathy《Yes you should understand backprop》** (blog)。
- **D2L (d2l.ai)** (book, 中英双语免费)。

### 动手项目
1. **核心：纯 NumPy 可训练 MLP**（Linear/ReLU/Softmax-CE 三层各写 forward/backward，mini-batch SGD+动量，MNIST >97%，先梯度检查<1e-7）。
2. micrograd 风格自动求导引擎（解 XOR）。
3. 模块化组件消融（He init/BN/dropout/weight decay，对照曲线图）。
4. 初始化与激活诊断（6-8 层各激活值/梯度标准差直方图，看信号消失/爆炸/死亡 ReLU）。
5. PyTorch 优化器与调度消融（SGD/Momentum/RMSProp/Adam/AdamW × constant/step/cosine+warmup，写训练技巧消融报告，衔接阶段1 capstone）。

### 常见误区
不做梯度检验就训练；softmax/log 数值不稳；Dropout 训练/推理混淆（忘 inverted/eval）；BN 推理用错统计量/小 batch；初始化与激活不匹配；学习率没调就下结论；Adam 里用 L2 当 weight decay（应 AdamW）；梯度爆炸不裁剪/消失误判；只看训练损失不划验证集。

### 时间估计
5–6 周（60–80h）。第1周感知机/MLP/计算图；第2周反向传播手推+梯度检验；第3周核心 NumPy MLP；第4周激活/初始化/损失诊断；第5周正则化与归一化；第6周优化器/调度+PyTorch 消融报告。

### 掌握自检
能白纸推两层 ReLU+softmax 网络全部反向公式（矩阵形式）；能纯 NumPy 实现并通过梯度检查、MNIST>97%；能解释 XOR；能从方差守恒推 Xavier/He；能写 inverted dropout 与 BN 的 train/eval 两套逻辑；能解释 SGD/Momentum/RMSProp/Adam/AdamW 差异与 AdamW 必要性；面对"训不动"能给系统诊断清单；能解释 BN 加速效果与对初始化/学习率的放松。

---

# 阶段 2 · 视觉与序列：两大经典架构家族（8–11 周）

> 目标：掌握处理网格数据的 CNN 与处理序列数据的 RNN/LSTM/GRU，理解归纳偏置如何随数据结构变化，第一次接触注意力雏形与编码器-解码器——这是通往 Transformer 的关键跳板。M6/M7 相对独立，快速路线可任选其一精读，**但 M7 的 attention 不可跳**。

## 模块 6 · 卷积神经网络与计算机视觉

**一句话**：从卷积运算的归纳偏置出发，串起经典 CNN 演进（LeNet→ResNet→DenseNet）、迁移学习与下游视觉任务，并理解为何注意力催生了 Vision Transformer，为 Transformer 枢纽埋伏笔。

### 学习目标
- 用"参数共享+局部连接+平移等变性"解释卷积的归纳偏置。
- 徒手计算卷积/池化输出尺寸、参数量、FLOPs 与感受野。
- 讲清 LeNet→ResNet 演进主线，从优化角度解释残差连接为何让超深网络可训练。
- PyTorch 从零搭 mini-ResNet 在 CIFAR-10 >90%，可视化卷积核/特征图。
- 正确执行迁移学习（冻结 vs 微调、分层学习率）。
- 说清分类/检测/分割三类任务的输入输出与代表架构。
- 阐述 CNN 局部偏置在数据充足时成瓶颈，从而理解 ViT 动机。

### 先修
M5 深度学习基础；M3 PyTorch 工程；M1 线代/链式法则；基础图像概念（H×W×C 张量、RGB、归一化）。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **卷积本质与归纳偏置** `[全]` | 卷积=互相关；局部连接/参数共享/平移等变（区分等变 vs 不变）；多通道参数量 C_out×C_in×k×k；输出尺寸公式；1×1 卷积 | 视觉深度学习的原子操作 |
| **感受野/池化/下采样** `[全]` | 感受野递推；堆叠小核（两层3×3≈5×5）；max/avg pooling；stride 卷积 vs 池化；全局平均池化；空洞卷积 | 连接"局部卷积"与"全局理解"的桥梁 |
| **经典演进 I：LeNet/AlexNet/VGG** `[全]/[快略]` | LeNet 模板；AlexNet（ReLU/Dropout/GPU）；VGG（小核加深但参数爆炸） | 展示"加深"逼近极限并暴露两大难题 |
| **经典演进 II：GoogLeNet/ResNet/DenseNet** `[全]` | Inception 多尺度+1×1 bottleneck；**ResNet 残差块 F(x)+x**（从优化角度解释梯度高速公路与退化问题）；DenseNet 拼接 | 残差思想成为 Transformer/扩散标配，连接 M9 |
| **迁移学习与微调** `[全]` | 浅层通用/深层任务特定；特征提取 vs 微调；分层学习率+ImageNet 归一化+增强；何时失效 | 视觉工程默认起点，预演"预训练+微调" |
| **下游任务：分类/检测/分割** `[全]/[快略]` | 检测两/单阶段（R-CNN/YOLO，IoU/NMS/mAP）；分割（FCN/U-Net/DeepLab）；共享 backbone+任务头；编码器-解码器 | 检测/分割是工业落地主力 |
| **从 CNN 到 ViT 动机** `[全]` | 归纳偏置双面性；ViT patch→token+位置编码+全局自注意力；数据规模决定 CNN vs ViT；Swin/ConvNeXt | 阶段2 通往阶段3 的桥梁 |

### 关键资源
- **CS231n** (course) — CNN 入门金标准，作业用 NumPy 从零实现卷积。
- **D2L 第7–8章** (book) — PyTorch 逐一实现经典 CNN。
- **AlexNet/VGG/ResNet/DenseNet/ViT 原论文** (paper) — **ResNet 务必精读**。
- **PyTorch Transfer Learning Tutorial + torchvision.models** (blog)。
- **CNN Explainer (poloclub)** (blog) — 交互可视化。
- **labml.ai 注释实现 / timm** (repo)。

### 动手项目
1. NumPy 手写 2D 卷积与池化（Sobel 边缘检测 + 反向传播对比 autograd）。
2. 感受野与尺寸计算器（分析 VGG-16/ResNet-18）。
3. **阶段2 capstone 核心：mini-ResNet 训 CIFAR-10 >90%**（含 BN/增强/调度/TensorBoard；消融去残差/加深到 56 层复现退化现象）。
4. 可视化卷积核与特征图。
5. 迁移学习实战（冻结 vs 微调对比）。
6. （选做）跑通现成 Faster R-CNN/DeepLabV3 推理，建立任务直觉。

### 常见误区
卷积输出尺寸算错；混淆等变 vs 不变；以为是数学卷积（实为互相关）；忘归一化/用错 mean-std；BN 小 batch/推理用错；微调学习率过大冲毁预训练；盲目堆深不加残差/BN；CIFAR 直接套 224×224 架构；认为 ViT 全面碾压 CNN。

### 时间估计
4–5 周（50–70h）。工程快速路线压到 2.5–3 周（侧重 capstone+迁移学习）。

### 掌握自检
能写卷积输出尺寸公式并徒手算尺寸/参数量/感受野；能解释三大归纳偏置并区分等变/不变；能 NumPy 写 2D 卷积前向并与 PyTorch 对拍；能从优化角度解释残差让 100+ 层可训练 + 说清退化实验；能搭 mini-ResNet 训到 >90% 并复现退化；能正确迁移学习并解释策略选择；能说清三任务输入输出+共享 backbone；能讲清 ViT 与数据规模权衡。

---

## 模块 7 · 循环网络与序列建模（RNN / LSTM / GRU）

**一句话**：从时序依赖与 RNN 的展开/BPTT 出发，吃透梯度消失与门控（LSTM/GRU）的缓解机制，再走通 seq2seq 编码器-解码器与注意力雏形——最终从信息流与并行性两个角度论证 RNN 为何被 Transformer 取代，为 M9 铺设最强跳板。

### 学习目标
- 解释序列数据归纳偏置（时间参数共享、变长、顺序敏感）。
- 把 RNN 沿时间展开成共享权重深层计算图，手推 BPTT 并解释梯度连乘。
- 从雅可比连乘定量解释梯度消失/爆炸；掌握梯度裁剪。
- 从零实现 vanilla RNN 与 LSTM cell 前向/反向，解释门控与 cell state 加法更新为何缓解梯度消失。
- 说清 GRU 相对 LSTM 的简化与工程取舍；解释双向 RNN 适用与因果限制。
- 画出 seq2seq 信息流，指出定长上下文向量瓶颈，解释 Bahdanau/Luong 注意力。
- 用"信息流"和"可并行性"精确说出 RNN 两个根本瓶颈，论证 Transformer 设计动机。

### 先修
M5（反向传播/计算图/优化/梯度问题）；M1/M2（雅可比/链式法则/softmax/交叉熵）；M3 熟练 PyTorch；M8 嵌入基本概念（仅用到"离散 token→向量"）；**最小分词概念（吸收评审，见下）**。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **最小分词概念（前置，新增）** `[全]` | 字符级 vs 词级 vs 子词的存在；本模块用字符级即可；为何 OOV 是问题 | **吸收评审：避免分词在 M9 才迟到，字符级 LM capstone 需要它** |
| **序列数据与时序依赖** `[全]` | 变长+依赖+顺序；语言建模 P(x)=∏P(xₜ\|x_<t)；**困惑度=exp(平均交叉熵)**；hidden state=被压缩的历史；任务形态分类；teacher forcing 与 exposure bias | 整个大模型时代本质是自回归 LM |
| **RNN 展开与 BPTT** `[全]` | h_t=tanh(W_hh h_{t-1}+W_xh x_t)；展开成深度=T 前馈网络；∂h_t/∂h_k=∏雅可比；truncated BPTT+detach；正交初始化 | 理解 RNN 一切问题的数学入口 |
| **梯度消失与爆炸** `[全]` | 雅可比谱半径<1 消失/>1 爆炸；tanh/sigmoid 饱和；爆炸→NaN 用裁剪治，消失更隐蔽；加法式状态=梯度高速公路 | 解释 vanilla RNN 记不住长程依赖 |
| **LSTM：门控与 cell state** `[全]` | C_t=f_t⊙C_{t-1}+i_t⊙C̃_t（加法通路）；遗忘/输入/输出门；f_t≈1 时梯度不衰减；遗忘门偏置初始化为正 | 核心 implement-to-understand 里程碑 |
| **GRU：精简门控** `[全]` | 更新门+重置门，无独立 cell state；参数更少更快；无普适赢家 | "门控"是可裁剪的设计空间 |
| **双向与多层 RNN** `[全]` | 双向看左右上下文（编码类有益）；**双向不能用于自回归生成**；stacked RNN+层间 dropout；ELMo | 双向 vs 单向是 BERT vs GPT 的分水岭伏笔 |
| **seq2seq 编码器-解码器** `[全]` | 编码器压成定长 context vector，解码器自回归生成；**信息瓶颈**；贪心 vs beam search；teacher forcing；BLEU | 催生注意力的直接动因 |
| **★ 注意力起源 Bahdanau/Luong** `[全]` | 对齐权重+动态上下文向量；加性 vs 乘性打分；可解释热力图；与 self-attention 的 Q/K/V 对应 | **跳板中的跳板，M9 强先修，不可跳** |
| **RNN 根本瓶颈 → Transformer** `[全]` | 瓶颈一：顺序计算不可并行；瓶颈二：长程路径长度线性增长（self-attention 为 O(1)）；O(T²) 代价；时间线 | 全程转折锚 |

### 关键资源
- **CS224n** (course) — 序列建模黄金标准课程。
- **Karpathy《Unreasonable Effectiveness of RNNs》** (blog)。
- **Olah《Understanding LSTM Networks》** (blog) — LSTM 最清晰图解。
- **LSTM (1997) / GRU+seq2seq (2014) / Bahdanau (2015) / Luong (2015) 原论文** (paper)。
- **D2L RNN/注意力章节** (book) — 完整 PyTorch 从零实现。
- **Pascanu et al. (2013) 梯度消失/爆炸** (paper)。

### 动手项目
1. 从零 vanilla RNN（NumPy/张量，手写 BPTT，字符级莎士比亚，观察梯度指数衰减）。
2. **从零 LSTM cell**（手写四门 + cell state，对比 RNN，遗忘门偏置消融，核心里程碑）。
3. PyTorch nn.LSTM/GRU 复现 char-rnn/makemore（GRU vs LSTM 参数/速度/效果对比表）。
4. 不带 attention 的 seq2seq（日期格式归一化，长序列暴露瓶颈）。
5. 加 Bahdanau/Luong 注意力（对比 + 注意力热力图，通往 M9 关键一跃）。
6. RNN 串行瓶颈基准（单步前向耗时随 T 线性增长，为"RNN 为何被取代"提供实验证据）。

### 常见误区
混淆"展开深度"与"多层深度"；以为 LSTM 彻底解决梯度消失（只是缓解）；张量形状灾难（batch_first/双向翻倍）；梯度裁剪只治爆炸不治消失；自回归生成错用双向；忽视 exposure bias；把注意力当黑盒 trick；BPTT 忘记累加共享权重梯度/忘 detach 致显存爆炸；窗口短于依赖跨度学不到长程。

### 时间估计
3–4 周（35–55h）。工程快速路线压到 2 周（精读 LSTM 与 attention，**attention 不可跳**），字符级 LSTM LM 整理为阶段2 capstone 的一半。

### 掌握自检
能白板展开 RNN 并手推 ∂h_t/∂h_k 雅可比连乘、定量解释梯度消失/爆炸条件；能从零实现 LSTM cell 并解释 cell state 加法更新；能说清 GRU vs LSTM 三条工程取舍；能解释双向不能自回归→BERT vs GPT 区别；能画 seq2seq+attention 信息流并指出解决/未解决的痛点；能写 Bahdanau/Luong 打分函数并对应 Q/K/V；能用两条主线论证 RNN 瓶颈+Transformer 如何回应；能展示加 attention 前后差距+读懂热力图。

---

# 阶段 3 · 表示学习与 Transformer 枢纽（9–12 周）

> 全程的转折与核心。先通过 AE/VAE/GAN/嵌入建立"学习紧致表示与生成分布"视角，随后集中火力攻克 Transformer 与自注意力——它是 LLM、扩散、多模态的共同地基。**M9 是整个路线最高优先级，配合 capstone"从零实现 nanoGPT"，建议放慢节奏反复打磨。**

## 模块 8 · 表示学习与生成模型 I（AE / VAE / GAN / 嵌入 / 对比学习损失）

**一句话**：从"压缩重建"到"学习并采样一个数据分布"，建立表示学习与生成建模的统一视角：用自编码器理解瓶颈表示，用 VAE 打通隐变量与变分推断，用 GAN 理解对抗博弈，用词嵌入与对比学习理解"语义即几何"，为后续 Latent Diffusion 与多模态埋伏笔。

### 学习目标
- 区分判别式 vs 生成式建模；解释"学习紧致表示"与"学习数据分布"是一枚硬币两面。
- 从零实现 AE/去噪 AE/稀疏 AE，解释三种正则各逼模型学什么。
- 逐项推导 VAE 的 ELBO，讲清重参数化为何让采样可反传、KL 项的信息瓶颈双重作用。
- 解释 VAE 与信息论（KL、率失真、后验坍塌）、β-VAE 解耦。
- 写出 GAN minimax 目标，解释判别器最优时最小化 JS 散度，识别缓解模式崩溃。
- 解释 word2vec（Skip-gram+负采样）与 GloVe 目标差异、嵌入几何（king−man+woman≈queen）。
- 讲清对比学习核心（InfoNCE、正负样本、SimCLR 增强）—— **CLIP 的多模态对齐推迟到 M10**。
- 说出 VAE 在 Latent Diffusion 中的角色。

### 先修
M2 概率+信息论（高斯、KL、MLE、**Jensen/变分下界直觉**）+最优化；M3 PyTorch；M5 深度学习基础；M6（部分）卷积/转置卷积；**M9 的注意力非硬门槛**。

> **吸收评审（顺序修正）**：M8 **只讲对比学习损失的纯表示学习视角**（SimCLR/InfoNCE/嵌入几何），**CLIP 的图文多模态对齐移到 M9 之后（并入 M10）**，因为 CLIP 文本编码器依赖 Transformer。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **表示学习统一视角** `[全]` | 好表示=低维/解耦/对下游有用；判别 p(y\|x) vs 生成 p(x)；自监督；潜变量 z 的"真实自由度"直觉；脉络 AE→VAE→GAN→嵌入/对比→扩散 | 阶段3 世界观 |
| **自编码器 AE** `[全]` | 编码器/瓶颈/解码器+重建损失；瓶颈强迫丢冗余；线性 AE+MSE=PCA；**AE 隐空间不是生成式的**（随机 z 解码出噪声）；异常检测 | VAE 的动机来源 |
| **去噪 AE / 稀疏 AE** `[全]` | DAE 加噪重建干净→学数据流形；**DAE 隐式学 score ∇log p(x)（通往扩散）**；稀疏 AE（L1/KL 稀疏惩罚）；**稀疏 AE 是 2024+ LLM 可解释性核心工具（呼应 M-I）** | 三种正则注入归纳偏置；两条线连向前沿 |
| **★ VAE：ELBO 与重参数化** `[全]` | p(x)=∫p(x\|z)p(z)dz 不可解；变分推断 ELBO=重建−KL(q‖p)；**重参数化 z=μ+σ⊙ε**（采样可反传）；AE 编码"点"vs VAE 编码"分布"；KL warmup 防后验坍塌；VAE 偏模糊原因 | 连接深度学习与概率推断的枢纽，Stable Diffusion 的压缩器 |
| **VAE 与信息论** `[全]/[研]` | KL=额外码率→率失真框架；后验坍塌（码率压到0）；β-VAE（信息瓶颈强度 vs 重建）；互信息视角 | 把 M2 信息论用起来 |
| **生成对抗网络 GAN** `[全]` | minimax；判别器最优 D*=p_data/(p_data+p_g)→最小化 JS 散度；非饱和损失/WGAN-GP/谱归一化；模式崩溃识别与缓解；FID/IS 评估 | 隐式生成模型路线 |
| **词嵌入 word2vec/GloVe** `[全]` | 分布假说；Skip-gram+负采样；GloVe 共现矩阵分解；余弦相似度+线性类比（共现对数线性结构涌现）；静态 vs 上下文嵌入 | 离散世界接入网络的接口，"语义=几何"，RAG/向量检索基石 |
| **对比学习 SimCLR（仅损失视角）** `[全]` | InfoNCE=互信息下界（与负采样一脉）；SimCLR（强增强+投影头+大 batch）；MoCo/BYOL/DINO；对比 vs 生成式表示学习 | 2020 后自监督主流；**CLIP 留到 M10** |
| **承前启后：VAE 在 Latent Diffusion** `[全]` | 像素空间扩散算力爆炸→VAE 压到低维潜空间跑扩散；VAE=感知压缩器 | 把零散知识缝成通往前沿的网 |

### 关键资源
- **CS231n 生成模型一讲（AE/VAE/GAN）** (course)。
- **CS224n 词向量两讲（word2vec/GloVe）** (course)。
- **Lilian Weng《From Autoencoder to Beta-VAE》/《From GAN to WGAN》** (blog) — 跟着推 ELBO。
- **Auto-Encoding Variational Bayes (2013)** (paper) — VAE 原始。
- **GAN (2014) + NIPS 2016 GAN Tutorial** (paper)。
- **SimCLR (2020) + Lilian Weng《Contrastive Representation Learning》** (paper/blog) —（CLIP 移到 M10）。
- **word2vec (2013) / GloVe (2014)** (paper)。
- **pytorch/examples (vae/dcgan) + lucidrains** (repo)。
- **Latent Diffusion (Rombach 2022) 第3节前半** (paper) — 仅读 VAE 潜空间动机。
- **Embedding Projector + Jay Alammar《Illustrated Word2vec》** (blog)。

### 动手项目
1. **AE 三连**（普通卷积 AE / 去噪 AE / 稀疏 AE，2维隐空间散点 + 网格采样看"AE 隐空间不连续"——VAE 动机）。
2. **从零 VAE**（自写 ELBO+重参数化；随机采样生成 + 隐空间插值；KL annealing 实验防后验坍塌）。
3. **DCGAN 并复现一次模式崩溃**（训练日志+失败分析，再用 WGAN-GP/标签平滑缓解）。
4. 词嵌入几何探索（gensim word2vec 或 GloVe，验证类比 + t-SNE 可视化 + most_similar）。
5. （进阶）迷你 SimCLR（CIFAR-10 InfoNCE + 线性探针，对比预训练 vs 随机特征）。
6. （选做衔接）潜空间生成概念串联（固定 VAE 编码器，在 z 上训小生成模型，写笔记连接 Latent Diffusion）。

### 常见误区
把 KL 项当黑魔法死记（要手推 ELBO）；重参数化没理解透（随机性必须外置到 ε）；后验坍塌不自知（KL≈0 要警觉）；AE 与 VAE 混为一谈；GAN 把所有问题归为学习率（loss 不指示进展，要看样本）；模式崩溃当"没训好"硬训；图像 VAE 模糊以为代码错（原理性短板）；把类比当精确等式；对比学习忽视增强/大 batch；输出激活与损失不匹配。

### 时间估计
4–5 周（50–70h）。第1周表示学习+AE/DAE/稀疏；第2–3周 VAE（重头，放慢）；第3–4周 GAN；第4周嵌入+对比学习+Latent Diffusion 衔接。

### 掌握自检
能从 log p(x) 推 ELBO 并解释重建/KL 项与信息论含义；能讲清重参数化为何梯度可回传；能做隐空间插值并解释 VAE 能而 AE 不能；能写 GAN minimax 并证明判别器最优时最小化 JS+识别缓解模式崩溃；能对比 AE/DAE/稀疏正则各逼学什么+指出 DAE 与扩散 score 联系；能说清 word2vec vs GloVe 目标差异+复现类比；能写 InfoNCE+说清 SimCLR 正负例/增强/batch；能讲清"为何 Stable Diffusion 先用 VAE 压到潜空间"。

---

## 模块 9 · Transformer 架构与注意力（核心枢纽 ★最高优先级）

**一句话**：从零吃透自注意力、多头机制、位置编码与三种 Transformer 范式（BERT/GPT/T5），精读《Attention Is All You Need》，最终能逐行实现并解释一个 decoder-only 小型 GPT——这是连接前面所有架构与后面所有大模型前沿的分水岭模块。

### 学习目标
- NumPy 从零实现缩放点积注意力，解释张量形状随 batch/seq/头数变化。
- 解释 Q/K/V 角色分工，把注意力理解为"可微分软字典查找"。
- PyTorch 从零实现多头自注意力、因果掩码、残差+LayerNorm、FFN，组装完整 block。
- 对比绝对/相对/RoPE 位置编码，说清如何把顺序信息注入置换不变的注意力。
- 讲清三范式训练目标与适用（BERT+MLM、GPT+CLM、T5/encoder-decoder），解释为何主流走向 decoder-only。
- 解释 BPE/WordPiece/SentencePiece 原理与差异，手动跑通 BPE 合并。
- 逐节复述《Attention Is All You Need》并批判性指出已被改进之处。
- 搭建 decoder-only 小型 GPT（nanoGPT），训练并采样生成连贯文本。

### 先修
M5 深度学习基础；**M7 强先修**（seq2seq+attention 信息流、RNN 两瓶颈）；M8 嵌入；M1 线代（追踪 (batch, seq, d_model)）；M3 PyTorch（einsum/bmm）。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **从 RNN 瓶颈到注意力** `[全]` | 直接连接路径 O(1) vs RNN O(n)；自/交叉/掩码注意力 | 抓住"直接连接+可并行"主线 |
| **★ 缩放点积注意力** `[全]` | softmax(QKᵀ/√d_k)V 逐项拆解；**为何除 √d_k**（防 softmax 饱和）；张量形状全程追踪；掩码用 −∞ 加（非 0 乘）；软字典查找 | 唯一真正"新"的计算原语 |
| **自注意力与 Q/K/V** `[全]` | Q/K/V 来自同一 X 的独立投影；非对称相似度；**置换不变→必须加位置编码**；注意力可解释性争议 | 从套公式到懂设计的关键一跃 |
| **多头注意力** `[全]` | 多个相似度子空间；reshape 到 (batch, h, seq, d_k) 并行；concat+W_O；MQA/GQA 伏笔 | 表达力核心，reshape 最易出错处 |
| **位置编码：绝对/相对/RoPE** `[全]/[RoPE 深化留 M11/M14]` | sinusoidal/可学习绝对；相对（T5/ALiBi）；**RoPE 旋转**（实现简洁+外推优势，现代标准）；加法 vs 注入方式 | 长上下文的核心战场（**M9 简述 RoPE，M11/M14 结合长上下文深入**） |
| **残差+LayerNorm+FFN（完整 block）** `[全]` | Pre-LN vs Post-LN（现代用 Pre-LN）；残差=梯度高速公路；FFN（4×宽，GELU/SwiGLU）；**LayerNorm 沿特征维（M5 已讲 BN 基础，此处只差异化）**；RMSNorm | 把单一原语堆成可训练深层网络的胶水 |
| **范式一：Encoder-only BERT+MLM** `[全]/[快略]` | 双向；MLM 遮 15%；预训练-微调；[CLS]/[SEP]/[MASK]；不能直接生成 | 检索/嵌入/分类主力（RAG 检索器） |
| **范式二：Decoder-only GPT+CLM** `[全]` | 自回归 CLM；**因果掩码**（下三角）；为何主流（统一目标/ICL/可扩展/KV 缓存）；GPT 演进 | 所有主流 LLM 骨架，capstone 目标 |
| **范式三：Encoder-Decoder T5** `[全]/[快略]` | 交叉注意力（Q 自身，K/V 编码器）；T5 text-to-text；span-corruption；三范式对比表 | 理解交叉注意力最佳载体 |
| **分词 BPE/WordPiece/SentencePiece** `[全]` | 子词平衡词表/序列长；BPE 合并最高频对；byte-level BPE 永不 OOV；WordPiece 按似然；SentencePiece unigram；分词对数字/代码/多语言的隐性影响 | 模型与文本世界的接口（**M11 只讲规模化工程取舍，避免重复**） |
| **★ 精读《Attention Is All You Need》** `[全]` | 按节精读；第4节复杂度/路径长度对比表是精华；批判：Post-LN→Pre-LN、sinusoidal→RoPE、ReLU→GELU/SwiGLU | 培养"读论文—复现—质疑"品味 |

### 关键资源
- **Attention Is All You Need (2017)** (paper) — 必精读第3/4节。
- **Jay Alammar《Illustrated Transformer》** (blog) — 最佳可视化。
- **The Annotated Transformer (Harvard)** (blog) — 逐行 PyTorch。
- **CS224N / CS336** (course)。
- **Karpathy《Let's build GPT》视频 + nanoGPT/minGPT/minbpe repo** (video/repo) — capstone 直接配套。
- **RoFormer (RoPE, 2021) + EleutherAI《Rotary: A Relative Revolution》** (paper/blog)。
- **BERT/GPT-2/T5 原论文** (paper)。
- **Hugging Face NLP Course Tokenizers 章节** (course)。
- **Lilian Weng《The Transformer Family v2》** (blog) — 衔接 M14。

### 动手项目
1. NumPy 缩放点积注意力（含因果掩码，手算验证 + 形状打印）。
2. PyTorch MultiHeadSelfAttention（reshape 多头 + 形状断言 + 与官方对拍）。
3. 完整 Transformer block（Pre-LN vs Post-LN 对比稳定性）。
4. **★ Capstone：从零 nanoGPT**（token+位置嵌入 + N 个因果掩码 block + 输出投影，tiny-shakespeare 训练采样，逐行解释张量形状）。
5. minbpe 从零实现 byte-level BPE（train/encode/decode，与 tiktoken 对比）。
6. 位置编码实验（sinusoidal/可学习/RoPE 替换对比 + 外推）。
7. Annotated Transformer 复现 encoder-decoder（toy 翻译/序列反转，观察交叉注意力）。
8. 对比实验（同语料 RNN/LSTM vs nanoGPT，loss/质量/单步耗时，体会并行性）。

### 常见误区
忘除 √d_k 或缩放写错（不报错难发现）；因果掩码方向错/没在 softmax 前应用（信息泄露）；多头 reshape/transpose 维度错；掩码用 0 乘而非 −∞ 加；忽视位置编码（退化成 bag-of-words）；混淆 LayerNorm/BatchNorm；把注意力权重当可靠因果解释；nanoGPT 数据/标签错位一位；Post-LN 不加 warmup 致发散；只看视频不亲手敲；分词字节级细节没处理；d_model 不被 h 整除。

### 时间估计
**5–7 周（60–90h），全程最高优先级，宁可放慢反复打磨。** 第1周缩放点积注意力+NumPy；第2周多头+完整 block；第3周位置编码+残差/LN/FFN；第4周三范式+分词；第5周精读论文+Annotated 复现；第6–7周打磨 nanoGPT+拓展实验。

### 掌握自检
能白纸默写缩放点积注意力公式+解释 √d_k+softmax 维度+张量形状；能 PyTorch 从零写支持掩码的多头自注意力并对拍；能画完整 block 数据流（残差/LN 位置/FFN）+说清 Pre/Post-LN；能默画三范式对比表；能解释因果掩码+KV 缓存加速；能手动跑 BPE 合并+说清三分词差异+byte-level 为何永不 OOV；能讲清三种位置编码+RoPE 为何成标准；能逐节复述论文第4节+指出 3 处已改进设计；**完成 nanoGPT 训到 loss 下降并采样通顺文本，逐行解释每个张量**；能 5 分钟向只懂 RNN 的人讲清 Transformer 解决了什么。

---

# 阶段 4 · 生成模型 II 与大模型训练范式（9–12 周）

> 在 Transformer 地基上理解现代生成式 AI 两条主线：扩散+多模态生成，以及 LLM 预训练/缩放定律/分布式训练。理解"预训练学知识、海量算力换能力"的范式。研究路线建议两者都深入，工程路线可侧重 M11。

## 模块 10 · 扩散模型与多模态生成（含 CLIP）

**一句话**：在 Transformer 与 VAE/GAN 地基上，吃透扩散模型从前向加噪到反向去噪的数学与采样原理，理解 Stable Diffusion 等潜空间扩散的工程架构，打通 CLIP/VLM 的多模态对齐思路，最终从零实现一个可采样的 DDPM。

### 学习目标
- 推导前向加噪、闭式采样 q(x_t\|x_0)、反向去噪训练目标，解释为何简化为"预测噪声"MSE。
- 从零实现 DDPM（噪声调度+时间嵌入 U-Net+训练/采样循环），MNIST/CIFAR-10 生成。
- 解释扩散与 score-based、SDE 三视角统一（score=−噪声/σ）。
- 说清 DDIM 等快速采样、classifier-free guidance 机制与引导强度权衡。
- 画出 Stable Diffusion 数据流（文本→CLIP→交叉注意力→VAE 潜空间 U-Net→解码）。
- **解释 CLIP 对比学习如何对齐图文（吸收评审：CLIP 从 M8 移到此处，因依赖 Transformer 文本编码器）**。
- 系统对比扩散 vs GAN vs VAE（**统一对比表集中在此，消除重复**）。
- 概念性说清 ControlNet、图生图/重绘、文生视频与 VLM。

### 先修
M2 概率（高斯/KL/贝叶斯/MLE）；M8 VAE 的 ELBO+重参数化、GAN、潜空间、**对比学习损失**；**M9 Transformer/自/交叉注意力（CLIP 文本编码器、Stable Diffusion 条件、VLM 都依赖）**；M6 U-Net；M3 PyTorch。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **前向加噪与闭式采样** `[全]` | 固定无参马尔可夫链；重参数化合并 x_t=√ᾱ_t·x_0+√(1−ᾱ_t)·ε；t→T 趋纯高斯；linear/cosine 调度 | 整个框架的"锚"，训练目标的数学核心 |
| **★ 反向去噪与 DDPM 训练目标** `[全]` | 反向近似高斯；从 ELBO（与 VAE 同源）展开成 KL 项；**简化为预测噪声 MSE L=E‖ε−ε_θ‖²**；正弦时间嵌入；ε/x_0/v 三种参数化 | 模块数学心脏，capstone 理论依据 |
| **Score-based 与 SDE 统一** `[全]/[研]` | score=∇log p；NCSN+朗之万；**score≈−ε_θ/√(1−ᾱ_t)**；SDE/概率流 ODE | 理解 DDIM/DPM-Solver 合理性 |
| **采样器 DDPM→DDIM 加速** `[全]` | DDPM ~1000 步慢；DDIM 确定性大步跳（20–50 步）；DPM-Solver/UniPC（10–20 步）；蒸馏/consistency/LCM（1–4 步）；质量-速度-多样性三角 | 从研究走向产品的关键瓶颈 |
| **条件生成与 Classifier-Free Guidance** `[全]` | ε_θ(x_t,t,c)；CFG 训练随机丢条件+采样外推；引导强度 w（保真度 vs 多样性） | 现代文生图出图质量秘密武器 |
| **潜空间扩散与 Stable Diffusion** `[全]` | VAE 压到潜空间（算力降一个数量级）；三组件（VAE/U-Net/文本编码器）；交叉注意力注入文本；SDXL/SD3(DiT+rectified flow)/FLUX | 开源文生图事实标准 |
| **文生图/图生图/ControlNet/文生视频** `[全]/[快略]` | img2img（strength）；inpainting；**ControlNet 零卷积**；IP-Adapter/LoRA/DreamBooth；文生视频时间注意力（Sora/DiT） | 最有产品价值的部分 |
| **★ CLIP 与多模态对齐（从 M8 移入）** `[全]` | 对称 InfoNCE 在 4 亿图文对；图文共享嵌入空间；零样本分类；CLIP 在文生图的双重角色；SigLIP | **连接视觉与语言的桥梁，文生图/VLM 共同基石** |
| **视觉-语言模型 VLM** `[全]/[快略]` | 视觉编码器+连接模块（投影 MLP/Q-Former/交叉注意力）+LLM；视觉指令微调（LLaVA）；扩散=文→图 vs VLM=图→文；幻觉/计数难点 | 当下多模态主战场 |
| **★ 扩散 vs GAN vs VAE 系统对比（统一对比表）** `[全]` | 训练稳定性/样本质量/模式覆盖/推理速度/似然/潜空间；VAE 与扩散共享 ELBO 框架；latent diffusion=VAE+扩散 | **吸收评审：三处生成模型对比集中于此** |

### 关键资源
- **DDPM (Ho et al., 2020)** (paper) — capstone 蓝本。
- **Lilian Weng《What are Diffusion Models?》** (blog) — 最佳数学综述伴读。
- **The Annotated Diffusion Model (HF)** (blog) — 逐行 PyTorch 脚手架。
- **DDIM (2021) / Score-based SDE (2021) + Yang Song 博客** (paper/blog)。
- **Latent Diffusion (Rombach 2022)** (paper) — Stable Diffusion 源论文。
- **Classifier-Free Guidance (Ho & Salimans 2022)** (paper)。
- **CLIP (Radford 2021)** (paper) — 多模态对齐基石。
- **ControlNet (Zhang 2023)** (paper)。
- **Hugging Face diffusers 库** (repo)。
- **EDM (Karras) + MIT 6.S184/Flow Matching 课程** (course/paper)。
- **LLaVA (2023) + BLIP-2 (2023)** (paper) — VLM 通用范式。

### 动手项目
1. **核心 capstone：从零 DDPM**（纯 PyTorch，cosine 调度 + 时间嵌入 U-Net + 噪声预测 MSE + DDPM 采样，MNIST 清晰数字，再 CIFAR-10）。
2. 加 DDIM 采样（1000 步 vs 20/50 步质量/耗时曲线）。
3. 条件 DDPM + CFG（按标签生成，扫描 w 观察保真度/多样性）。
4. diffusers 跑通 Stable Diffusion 全流程（文生图/图生图/inpainting/ControlNet，架构图对应代码）。
5. 从零 mini-CLIP（小图像+文本编码器，对称 InfoNCE，零样本检索）。
6. （进阶）潜空间扩散迷你版（小 VAE 潜空间跑 DDPM，对比算力收益）。
7. （研究路线）复现一个消融（linear vs cosine 调度 / ε vs v-prediction）。

### 常见误区
把前向过程当需学习的网络；时间步 t 没作为条件/时间嵌入写错；闭式采样系数写反/索引错位；采样循环 t=0 仍加噪声；CFG 忘训练阶段丢条件；guidance scale 一味调大；彩色数据不归一化；误以为 SD 在像素空间扩散；训练步数预期不切实际（先 MNIST 验证管线）；把扩散与 GAN/VAE 对立看（SD=VAE+扩散）；CLIP/对比学习用过小 batch；只会调 diffusers 高层 API。

### 时间估计
4–6 周（50–80h）。第1–2周吃透 DDPM 数学+从零实现；第3周 score/SDE/DDIM/CFG；第4周 Latent Diffusion+diffusers；第5–6周 CLIP/VLM/文生视频+三范式对比+mini-CLIP。研究路线+1–2 周复现。

### 掌握自检
能推闭式采样并解释系数来历；能从 ELBO 推到简化 MSE 并说清"预测噪声=预测 score";能从零写 DDPM（时间嵌入 U-Net+训练/采样）MNIST 生成并逐行解释；能解释 DDIM 为何 20–50 步并实现对比；能画 SD 数据流说清各组件职责+为何潜空间；能解释 CFG 训练/采样两端+w 权衡；能讲清 CLIP 对比学习如何对齐图文+被文生图/VLM 复用；能一句话概括 VLM 架构+举代表模型；能从五维度对比扩散/GAN/VAE+解释 latent diffusion=VAE+扩散；能讲清 ControlNet 零卷积。

---

## 模块 11 · 大语言模型预训练与扩展（数据工程一等公民）

**一句话**：理解"基座模型"如何从海量文本中通过自监督预训练诞生：从训练目标、数据流水线、缩放定律、分布式并行到混合专家与评测，建立"预训练学知识、算力换能力"的工程化范式认知。

### 学习目标
- 解释预训练 vs 微调分工，说清 CLM/MLM/去噪三目标的损失、架构与代表模型。
- 描述完整预训练数据流水线（抓取→过滤→PII→精确/模糊去重 MinHash→配比→tokenizer），解释每步影响。
- 从零训练 BPE tokenizer（**规模化工程取舍，呼应 M9 已讲算法**）。
- 写出 Kaplan/Chinchilla 缩放定律，推导 C≈6ND 下 N/D 最优配比（~20 tokens/参数），判断"训练不足"还是"参数过大"。
- 区分数据/张量/流水线并行，解释 ZeRO 三阶段与 FSDP，估算 GPU 数与显存。
- 解释 MoE 路由/稀疏激活/负载均衡，说清固定 FLOPs 下放大参数及代价。
- 解释上下文长度对注意力的二次方影响与长上下文训练做法。
- 用困惑度与下游基准评估基座，理解数据污染。

### 先修
**M9 强先修**（从零 decoder-only block）；M5（交叉熵/AdamW/warmup+cosine/梯度裁剪/混合精度）；M2（MLE/熵/交叉熵/幂律）；M3 PyTorch；已完成 nanoGPT。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **预训练范式：自监督与基座诞生** `[全]` | 自监督=数据自身构造监督；基座只会续写，对话来自后训练；CLM/MLM/去噪对照；为何主流 decoder-only+CLM；"预训练学知识、后训练学对齐" | 总纲世界观 |
| **自监督目标数学与实现** `[全]` | CLM 损失+困惑度=exp(平均CE)；MLM 15% 遮挡样本效率低；T5 span corruption；**target=input 右移一位**；文档打包+跨文档 mask | 决定模型学到什么 |
| **Tokenization（规模化工程）** `[全]` | byte-level BPE 永不 OOV；词表大小权衡（32K–256K）；数字/代码/多语言压缩率；glitch tokens；压缩率=有效上下文/成本 | **M9 讲算法，此处只讲规模工程取舍（消除重复）** |
| **★ 数据流水线（一等公民，吸收评审）** `[全]` | 来源（CC/Wiki/书/代码/The Pile/FineWeb）；质量过滤（规则+分类器+困惑度）；**精确+模糊去重 MinHash/LSH**（降记忆/提样本效率）；**数据配比**（高质量上采样）；去污染；PII/安全；课程化退火；**合成数据 synthetic data**；**SFT/偏好数据构造方法学**；**数据版权与许可** | "数据 > 架构"是 2024-26 共识 |
| **★ Scaling Laws** `[全]` | L 随 N/D/C 幂律；**C≈6ND**；Kaplan（偏参数）vs **Chinchilla（~20 tokens/参数）**；训练最优 vs 推理最优（Llama 过度训练）；**数据受限缩放、推理时缩放、蒸馏缩放（吸收评审，新范式）**；局限（不预测涌现能力） | LLM 时代工程指南针 |
| **分布式训练** `[全]/[快略细节]` | 7B 需 ~100GB+；DP/TP/PP 切分维度；**ZeRO 三阶段/FSDP**（消除显存冗余）；Megatron TP；PP micro-batch；3D 并行+activation checkpointing+混合精度；通信瓶颈（TP 节点内、PP/DP 节点间） | "有想法"到"真能训出来"的鸿沟 |
| **混合专家 MoE** `[全]` | FFN→多专家+Top-k 路由（稀疏激活）；**细粒度+共享专家（DeepSeek）**；负载均衡（辅助损失/**无辅助损失偏置调整**）；Switch/Mixtral/DeepSeek-V3；激活参数定推理成本、总参定容量 | 前沿模型突破"参数=算力"瓶颈 |
| **上下文长度/成本/评测** `[全]` | 注意力 O(L²)+KV 缓存线性；两阶段训练+RoPE 外推（**位置插值/YaRN/NTK-aware**）+FlashAttention+ring attention；C≈6ND+MFU 估成本；loss spike 处理；困惑度局限；**下游基准 MMLU/HellaSwag/GSM8K/HumanEval + 污染**（呼应评测支线 M-E） | 落到可计算/可验证/可预算的现实约束 |

> **吸收评审（时效性）**：缩放定律纳入数据受限/推理时/蒸馏缩放；MoE 纳入细粒度+共享+无辅助损失均衡；位置编码外推纳入 YaRN/位置插值/NTK-aware。

### 关键资源
- **Stanford CS336（从零造 LM）** (course) — 最对口主线。
- **Kaplan (2020) + Chinchilla (Hoffmann 2022)** (paper) — 缩放对照。
- **GPT-3 (2020)** (paper)。
- **Llama / Llama 2 / Llama 3 技术报告** (paper) — 工程细节最丰富的真实案例。
- **Karpathy《Build the GPT Tokenizer》+ nanoGPT/build-nanogpt** (video/repo)。
- **Lee et al. (2021) 去重** (paper)。
- **The Pile/RedPajama/Dolma/FineWeb 论文与博客** (blog) — FineWeb 尤其可复现。
- **DeepSpeed ZeRO + Megatron-LM + PyTorch FSDP 文档** (paper)。
- **Switch/Mixtral/GShard** (paper)。
- **HF《Ultra-Scale Playbook》** (blog) — 大规模训练实战。
- **Eleuther lm-evaluation-harness** (repo) — 评测标准库。

### 动手项目
1. 从零 BPE tokenizer（minBPE，对比不同词表压缩率，观察数字/中文切分）。
2. **复现 GPT-2 (124M) 预训练**（build-nanogpt，FineWeb-Edu 子集，warmup+cosine+bf16+梯度累积，HellaSwag 对比）。
3. 迷你 scaling law 实验（5M/15M/50M/150M 参数 loss vs 参数对数图，拟合幂律 + Chinchilla 小规模验证）。
4. mini 数据流水线（语言/质量过滤 + MinHash 去重，对比去重前后困惑度）。
5. 从零 MoE FFN 层（8 专家 Top-2 路由 + 负载均衡损失，统计专家负载）。
6. FSDP 多卡训练（对比单卡 vs FSDP 显存，理解 ZeRO-3 分片；无多卡则估算 7B 显存账）。
7. lm-evaluation-harness 评测（HellaSwag/ARC/MMLU few-shot + 污染检查）。

### 常见误区
把"基座会聊天"当预训练产物；用 Kaplan"越大越好"指导现代实践（应 Chinchilla）；忽视 tokenizer 长期影响；跳过去重；评测不去污染；混淆激活参数与总参数（MoE）；低估分布式工程复杂度；只看困惑度判断好坏；过小 batch/忽略 warmup 训大模型。

### 时间估计
4.5–6 周。自监督+tokenizer 约1周；数据流水线约1周；scaling laws+迷你实验约1周；分布式+ZeRO/FSDP 约1–1.5周；MoE+长上下文/评测约1周；GPT-2 复现贯穿。研究路线 MoE/scaling +1–2 周。

### 掌握自检
能白板写 CLM/MLM/去噪损失+说清各自上下文/预测目标+为何 decoder-only+CLM；能从零实现 byte-level BPE+解释与 WordPiece/Unigram 差异+词表权衡；给定 C 能用 C≈6ND+~20 tokens/参数推 N/D，判断真实模型训练不足/过度训练+解释 Llama 过度训练；能画 DP/TP/PP+说清 ZeRO/FSDP 分摊+粗算 7B 显存；能解释 MoE Top-k 路由+负载均衡+总参 vs 激活参数；完成 GPT-2 复现展示 loss/困惑度曲线+HellaSwag 分数+说明去污染；能端到端复述基座诞生流程+指出每步易错处。

---

# 阶段 5 · 对齐、微调应用与前沿追踪（10–14 周 + 长期）

> 完成从"会预训练"到"会用好并对齐"的最后一跃：理解后训练与对齐如何把基座变成有用/无害/诚实的助手；掌握 PEFT/LoRA、Prompt、RAG、Agent、推理优化等落地技术；最后建立持续追踪 2024–2026 前沿的能力与品味。M14 转入长期持续追踪（每周固定 2–3 小时读论文/复现，无终点）。

## 模块 12 · 后训练与对齐（SFT / RLHF / DPO）

**一句话**：理解后训练如何把只会"续写"的基座变成有用、无害、诚实的对话助手：从 SFT 到 RLHF 的奖励模型+PPO 三段式，再到把偏好优化简化为分类损失的 DPO 及变体（IPO/KTO/ORPO），贯穿理解 KL 约束、奖励黑客与对齐本质。

### 学习目标
- 讲清"预训练→后训练"范式转变，说明为何纯预训练模型不能直接当助手。
- 从零写 SFT 训练循环并正确实现 loss masking（只在回答 token 算损失）。
- 画出 RLHF 三阶段，写出奖励模型 Bradley-Terry 损失，解释 PPO 目标里 KL 惩罚的来源与作用。
- 从"KL 约束下最大化奖励"最优解手推 DPO 损失，对比 RLHF vs DPO 目标函数差异。
- 说清 DPO 局限及 IPO/KTO/ORPO 变体动机。
- 区分 RLHF/RLAIF/Constitutional AI；用 HHH 框架分析对齐失败。
- 在偏好数据上跑通 DPO 微调并与 SFT 基线对比胜率。

### 先修
M11（预训练目标/tokenizer/解码）；M9（decoder-only 前向）；**M2 的强化学习速成（MDP/策略梯度/actor-critic/GAE/PPO，吸收评审消除断层）**；M2 概率/KL/MLE；M3 PyTorch+HF；M13 的 PEFT/LoRA 并行了解（消费级 GPU 几乎必用）。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **什么是后训练** `[全]` | 基座只学 P(next token)=超强自动补全；后训练=SFT+偏好对齐；能力（预训练）vs 对齐（后训练引出）；base→instruct/chat；chat template；InstructGPT 1.3B 胜 175B | 整个模块世界观 |
| **SFT / 指令微调** `[全]` | SFT=指令-回答对上自回归；**loss masking**（prompt 设 -100）；指令微调泛化；**数据质量>>数量**（LIMA 1000 条）；SFT 局限（只能模仿一个答案）；多轮打包/EOS | 后训练第一步，80% 场景性价比最高 |
| **人类偏好与奖励模型** `[全]` | 成对偏好 (prompt,chosen,rejected)；RM=标量头；**Bradley-Terry 损失 −log σ(r_w−r_l)**；RM=代理人类；RM 脆弱性→reward hacking/Goodhart；HH-RLHF | DPO 推导前提，理解 KL 约束必要性 |
| **★ RLHF 与 PPO** `[全]` | 三阶段（SFT→RM→PPO）；目标 max E[r]−β·KL(π‖π_ref)；**KL 约束防 reward hacking**（呼应 M2）；PPO 把生成建模成 token 级决策+裁剪；显存放 4 个模型；REINFORCE 类（RLOO/GRPO）简化 | 把 M2 KL 落到实处，理解对齐难/贵/不稳 |
| **★ DPO** `[全]` | KL 约束最优解 π*∝π_ref·exp(r/β)→反解 r=β·log(π/π_ref)；代回 Bradley-Terry **消掉奖励模型**；DPO 损失；离线/监督/单阶段/稳定；β=KL 强度；DPO 已知问题（降 chosen 绝对概率） | 开源对齐事实标准，智识高潮，必须手推 |
| **DPO 变体 IPO/KTO/ORPO** `[全]/[研]` | IPO（防确定性过拟合）；**KTO（非成对好/坏标签，前景理论，贴近点赞/点踩）**；ORPO（单阶段无 ref）；SimPO/cDPO；选型口诀 | 2024 后真实工程决策 |
| **RLAIF 与 Constitutional AI** `[全]` | AI 充当偏好标注者；CAI 两阶段（自我批评-修正+RLAIF）；把"什么是无害"从海量标注变少量原则；与红队关系；通往可扩展监督 | Claude 的训练范式，可扩展对齐方向 |
| **★ HHH、安全与红队（升级，吸收评审）** `[全]` | HHH 三维张力；诚实/真实性/幻觉；**谄媚 sycophancy**（RLHF 副作用）；**红队/越狱/prompt injection**；对齐评测（win rate/AlpacaEval/MT-Bench，长度偏置）；对齐税 | **从"怎么训"升华到"训成什么样、怎么验证"，呼应安全支线 M-S** |

> **吸收评审（升级安全）**：本模块的安全内容与贯穿支线 **M-S（伦理/安全/负责任 AI）** 联动——偏见与公平、隐私与数据治理、版权合规、AI 治理（EU AI Act）在 M-S 系统化，本模块聚焦对齐相关的越狱/谄媚/红队。

### 关键资源
- **InstructGPT (Ouyang 2022)** (paper) — RLHF 三阶段奠基，读图1。
- **DPO (Rafailov 2023)** (paper) — 第4节推导必自推。
- **Constitutional AI (Bai 2022)** (paper)。
- **Llama 2 (2023) 第3节** (paper) — 工业级 RLHF 最佳实践。
- **HF TRL 库（SFT/DPO/PPO/KTO/ORPO Trainer）** (repo) — 实操首选。
- **HF Alignment Handbook（Zephyr 配方）** (repo)。
- **KTO/ORPO/IPO 论文** (paper)。
- **CS224N/CS336 RLHF 讲座** (course)。
- **HF《Illustrating RLHF》+ Chip Huyen《RLHF》** (blog)。
- **Anthropic HH-RLHF/UltraFeedback + Red Teaming 论文** (repo/paper)。

### 动手项目
1. **SFT 最小实现**（小 base 模型 + LoRA，手写 loss masking，对比去掉 masking 的退化）。
2. 奖励模型 + Bradley-Terry 损失（HH-RLHF/UltraFeedback，测偏好预测准确率 + 分数分布）。
3. **DPO 复现（capstone 对应）**（TRL DPOTrainer，与 SFT 基线算胜率 + margin 曲线）。
4. 手推+代码验证 DPO 损失（自写 vs TRL 对拍 <1e-4）。
5. DPO vs KTO vs ORPO 小对比（胜率/稳定性/显存/时长 + 解释 ORPO 无需 ref）。
6. 迷你 Constitutional AI 自我批评（纯推理构造 批评-修正 数据对）。
7. 红队 + 谄媚探针（越狱/引导错误观点/诱导有害，记录失败 + 缓解方案）。

### 常见误区
SFT 忘 loss masking（学生成问题）；chat template 不一致；以为 RLHF/DPO 能教新知识；跳过/敷衍 SFT 阶段；忽视 KL/β 设置（reward hacking 或语言崩坏）；DPO 只看 loss 不看绝对概率（同时压低 chosen）；RM/judge Goodhart 化（长度偏置）；把无害做过头变无用；完整 PPO 显存低估（4 模型）；谄媚/标注者偏差放大。

### 时间估计
3–4 周（35–55h）。第1周后训练范式+SFT；第2周 RM+RLHF/PPO+KL（数学密集放慢）；第3周 DPO 推导+复现+变体（重头+capstone）；第4周 RLAIF/CAI+HHH/安全+对比报告。工程快速 2 周（PPO 只读不实现）；研究 +1 周。

### 掌握自检
能白板画 RLHF 三阶段+写各损失+解释 PPO KL 惩罚来源与去掉后果；能从 π*∝π_ref·exp(r/β) 推 DPO 损失并解释奖励模型如何消掉；能写 SFT loss masking 逻辑+指出缺失 masking 的 bug；能对比 RLHF vs DPO 目标差异+两优势两局限；能各一句话概括 IPO/KTO/ORPO 解决 DPO 哪个问题+按数据条件选；能区分 RLHF/RLAIF/CAI 反馈来源+描述 CAI 自我批评机制；跑通 DPO 展示与 SFT 胜率对比+解读 margin 曲线；能用 HHH 分析谄媚/越狱案例+提数据层缓解。

---

## 模块 13 · 微调与应用（PEFT / Prompt / RAG / Agent / 推理优化）

**一句话**：把"会预训练、会对齐"的基座真正用起来：以最小算力做 PEFT（LoRA/QLoRA），用提示工程榨干能力，用 RAG 接外部知识，用 Agent 调用工具完成多步任务，并用量化/KV cache/FlashAttention/投机解码/vLLM 把推理成本压到可上线——把 LLM 从 demo 变成产品的工程总成。

### 学习目标
- 说清全参微调 vs PEFT 权衡，从"低秩假设"推导 LoRA：W'=W+(α/r)·BA。
- 消费级 GPU 用 LoRA/QLoRA 微调 7B 级模型，正确合并/部署 adapter。
- 掌握提示工程方法论（zero/few-shot/CoT/self-consistency/结构化输出），解释 ICL 为何"不更新权重也能学"。
- 从零搭生产 RAG（切块→嵌入→向量库→检索→重排→生成），用 RAGAS 评估诊断。
- 判断该用 RAG/微调/结合。
- 实现 Agent（function calling/ReAct/MCP），搭会调工具、自我纠错的最小智能体。
- 掌握推理优化全栈（量化/KV cache/FlashAttention/PagedAttention/连续批处理/投机解码），用 vLLM 部署。
- 做端到端成本/延迟/质量三角权衡分析。

### 先修
**M9 强先修**（self-attention/KV cache/causal mask）；M11（tokenizer/自回归/混合精度/参数量与显存）；M12（SFT 数据格式/chat template）；M3 PyTorch；M1/M2（矩阵秩/低秩分解/余弦相似度）；**可用 GPU**（≥16GB 或 Colab T4）。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **微调谱系总览** `[全]` | 全参（显存 12–16× 参数）vs PEFT（<1% 参数）；四流派（加法/重参数化/前缀/选择）；低内在维度；决策维度（显存/数据/多任务/零延迟） | 整个模块地图 |
| **★ LoRA 原理与实现** `[全]` | ΔW=BA（r 远小于 d,k）；h=Wx+(α/r)BAx；A 高斯/B 零初始化；参数从 70 亿降到几百万；r/α/target_modules/dropout；推理合并零延迟 | 2023 以来微调标准 |
| **QLoRA/Adapter/Prefix/Prompt Tuning** `[全]` | QLoRA（4-bit NF4+双重量化+分页优化器，单卡微调 33B/65B）；Adapter（推理串行延迟）；Prefix/Prompt（软提示）；DoRA；选型口诀 | QLoRA 让普通人微调大模型 |
| **提示工程与 ICL** `[全]` | ICL 不更新权重；zero vs few-shot（recency/majority label bias）；**CoT《Let's think step by step》**；self-consistency 投票；ReAct/ToT/Least-to-Most；结构化输出约束；与推理模型关系 | 零成本提升第一手段 |
| **★ RAG 一：嵌入/切块/向量库/检索** `[全]` | RAG 动机（知识冻结/幻觉/私有数据）；嵌入（BGE/E5，bi-encoder）；**切块策略+overlap（头号变量）**；向量库（FAISS/Chroma/Qdrant/pgvector，ANN HNSW/IVF）；**混合检索（dense+BM25，RRF）**；top-k 与上下文预算 | 企业落地最主流方案 |
| **★ RAG 二：重排/评估/诊断/RAG vs 微调** `[全]` | **bi-encoder 召回+cross-encoder 重排**（行业标准两段式）；查询改写（HyDE/multi-query/分解）；**RAGAS（faithfulness/relevancy/context precision-recall）分段评估**；失败模式（召回失败/lost-in-the-middle/无关块致幻/多跳）；RAG vs 微调决策框架；GraphRAG/agentic RAG | demo 易、靠谱难，分水岭 |
| **★ Agents：function calling/ReAct/MCP** `[全]` | 感知-推理-行动循环；function calling（JSON schema）；**ReAct（Thought-Action-Observation）**；记忆/规划/反思/终止；**MCP（AI 应用的 USB-C，解决 M×N 集成）**；框架地形（先手写再用框架）；可靠性挑战（错误累积/成本爆炸） | LLM 从聊天走向干活的关键 |
| **★ 推理优化一：量化/KV cache/FlashAttention** `[全]` | **推理是带宽受限**（每 token 读全部权重）；量化（INT8/INT4/GPTQ/AWQ）；KV cache（O(n²)→逐 token O(n)）；MQA/GQA 压缩；**FlashAttention**（IO 感知 tiling/重计算）；PagedAttention | "跑得起但用不起"的关键 |
| **★ 推理优化二：投机解码/连续批处理/vLLM** `[全]` | 投机解码（草稿模型并行验证，2–3×）；Medusa/EAGLE；**连续批处理**（动态加入/移出）；**vLLM**（PagedAttention+连续批处理+OpenAI 兼容 API）；SGLang/TGI/llama.cpp/Ollama；吞吐/延迟（TTFT/TPOT）/质量/成本四角 | 推理成本降一个数量级的工程杠杆 |

> **吸收评审（时效性 + MLOps/Agent 系统）**：推理优化纳入 **FlashAttention-3、prefill/decode 分离、KV cache 压缩（MLA）** 等 2025-26 部署主流；Agent 部分补充 **多 Agent 编排、记忆系统、工具学习可靠性与评测、安全沙箱**（与 M14 联动）；MLOps 闭环（服务监控/A-B 测试/漂移检测/回归测试/可观测性 tracing/guardrails）在"概览补充"小节给出框架。

### 关键资源
- **LoRA (2021) / QLoRA (2023)** (paper)。
- **HuggingFace PEFT 库** (repo)。
- **CoT (Wei 2022) + Self-Consistency (Wang 2022)** (paper)。
- **OpenAI/Anthropic Prompt Engineering 官方指南** (blog)。
- **RAG (Lewis 2020) + Lost in the Middle (Liu 2023)** (paper)。
- **RAGAS 框架** (repo)。
- **ReAct (Yao 2022) + MCP 官方文档** (paper/blog)。
- **FlashAttention (2022)/FA-2 + PagedAttention/vLLM (Kwon 2023)** (paper/repo)。
- **CS336** (course)。
- **Lilian Weng《Prompt Engineering》/《LLM Powered Autonomous Agents》** (blog)。
- **Sebastian Raschka《Build a LLM from Scratch》+ PEFT 博客** (book)。

### 动手项目
1. LoRA 从零最小实现（自写 LoRALinear，替换 nanoGPT 线性层）。
2. **QLoRA 微调 7B 开源模型（capstone 第一部分）**（单卡 ≤24GB，自构 SFT 数据，合并 adapter+vLLM 部署）。
3. 提示工程消融（GSM8K 子集，zero/few-shot/CoT/self-consistency 准确率对比报告）。
4. **从零生产级 RAG（capstone 第二部分核心）**（递归切块→BGE/E5→FAISS/Chroma→混合检索→重排→带引用生成，RAGAS 评估 + 诊断 3 种失败模式）。
5. 手写 ReAct Agent（不依赖 LangChain，接计算器/检索/代码执行，防死循环+重试，可选 MCP server）。
6. vLLM 部署与推理优化基准（开关量化/不同 batch 的吞吐/延迟，vLLM vs 朴素 HF generate）。
7. （整合 capstone）端到端 LLM 应用（QLoRA 模型+RAG+Agent+vLLM，作品集仓库）。

### 常见误区
把微调当万能药（多数先 prompt+RAG）；LoRA 超参乱设（r 过大过拟合）；SFT 数据/chat template 不一致；QLoRA 把基座也训练/合并出错；RAG 切块草率（头号原因）；RAG 只评最终答案不分段；只用向量忽视关键词与重排；Agent 过度工程化（先手写最小 ReAct）；忽视推理是带宽受限（优化错方向）；量化无脑最低 bit；KV cache 显存爆炸被忽视；评估全靠肉眼。

### 时间估计
7–9 周（80–120h）。PEFT+LoRA/QLoRA 约2.5周；提示工程约1周；RAG 约2周；Agent+MCP 约1.5周；推理优化+vLLM 约1.5周；整合 capstone 约1周。工程快速 5–6 周（精做 QLoRA+RAG+vLLM）；研究 +2 周。

### 掌握自检
能从低秩假设推 LoRA 公式+解释 A/B 初始化+α/r+合并；能 QLoRA 微调 7B 完成任务+正确处理 chat template+部署，效果优于基座；给定场景能论证用 prompt/RAG/微调/组合；能从零搭含混合检索+重排的 RAG+RAGAS 分段评估+3 种失败模式诊断修复；能手写 ReAct Agent+解释 MCP 解决什么；能解释推理为何带宽受限+各优化针对哪个瓶颈；能用 vLLM 部署+产出吞吐/TTFT/TPOT 对比表；能读 LoRA/QLoRA/RAG/ReAct/FlashAttention 任一原论文复述贡献与局限。

---

## 模块 14 · 前沿进展（2024–2026）：推理模型、MoE、长上下文、多模态、Agent 与模型版图

**一句话**：把 M1–13 的地基汇聚到当下，理解 2024–2026 真正改变格局的几条主线（用 RL 造出"会思考"的推理模型、用稀疏 MoE 把参数做大算力做省、把上下文拉到百万级、走向原生多模态与能操作计算机的 agentic 系统、世界模型/视频生成、蒸馏小模型），并建立一套能长期自我更新、读懂并复现最新论文的方法论与品味。

### 学习目标
- 用统一语言解释"测试时计算 test-time compute"范式与训练时缩放的关系。
- 讲清推理模型训练配方（o1/o3→DeepSeek-R1 纯 RL，GRPO/RLVR，冷启动→RL→拒绝采样蒸馏），指出长 CoT 何时有效/过度思考。
- 从架构解释稀疏 MoE（路由/Top-k/负载均衡/共享/细粒度专家）。
- 说清超长上下文（128K→1M+）实现要素与局限（needle ≠ 长程推理，RULER/LongBench）。
- 描述原生多模态/全模态范式转变（GPT-4o 原生图像、Gemini 原生多模态、理解+生成统一）。
- 刻画 agentic 系统能力边界，用真实基准（SWE-bench Verified/GAIA/OSWorld/τ-bench）判断 demo 与可靠落地差距。
- 概述世界模型/视频生成前沿，分清已实现与被夸大。
- 在高效化维度做工程取舍（蒸馏/量化/小模型）。
- **画出 2026 年中开源 vs 闭源版图的判断框架（而非写死型号）**，并建立可持续前沿追踪工作流。

### 先修
M9（self-attention/KV/位置编码）；M11（scaling law/并行）；M12（SFT/RLHF/DPO + RL 速成）；M13（agent/推测解码/vLLM/量化）；扎实英文论文阅读能力。

### 主题与要点

| 主题 | 关键点 | 为什么重要 |
|---|---|---|
| **范式总览：第二条缩放曲线** `[全]` | 训练时缩放遇数据墙；**test-time compute**（更长 CoT 换正确率）；三形态（串行长 CoT/并行投票 best-of-N/树迭代）；o1 算力对数线性提升；时间线 | 理解整个格局的总纲 |
| **推理模型 I：o1/o3 与长 CoT** `[全]` | 大规模 RL 优化隐藏推理链；reasoning_effort；数学/代码提升最大；自我反思/回溯涌现；过度思考；ARC-AGI 争议；快/慢双模型+thinking budget | 推向主流的引爆点（**区分官方声称 vs 推断**） |
| **★ 推理模型 II：DeepSeek-R1 开放配方** `[全]` | **R1-Zero 纯 RL（无 SFT）涌现长 CoT**；**RLVR**（可验证奖励）；完整流水线（冷启动 SFT→推理 RL→拒绝采样→再 SFT→对齐 RL）；**GRPO（去 critic，组相对奖励，省显存）**；推理蒸馏；open-r1/PRM vs ORM | **最该亲手复现的部分，把 M12 RL 打通到前沿** |
| **MoE 前沿** `[全]` | 多专家+Top-k 路由（稀疏激活）；**细粒度+共享专家（DeepSeek）**；**无辅助损失负载均衡**；DeepSeek-V3(671B/激活37B)/Mixtral/Qwen3/Llama-4；激活参数定成本、总参定容量；显存/通信代价 | 前沿大模型几乎一致选择 |
| **超长上下文** `[全]` | O(n²)+KV 缓存瓶颈；**RoPE 外推/位置插值/YaRN/NTK-aware**；Ring Attention/MLA/KV 压缩；**needle 通过≠长程多跳推理**（RULER/LongBench/lost-in-the-middle）；Gemini 1M；长上下文 vs RAG | 最易被数字游戏误导处 |
| **多模态/全模态/原生多模态** `[全]/[快略]` | 三代（拼接/原生/全模态 omni）；统一 token 化+early fusion；**GPT-4o 原生图像生成（理解+生成统一）**；端到端语音；长视频理解 | 通向通用感知-行动智能体 |
| **Agentic 系统与计算机使用** `[全]` | 工具+规划+记忆+反馈循环；**computer use（Claude/Operator）**；**SWE-bench Verified**（自主修 bug）；真基准（GAIA/OSWorld/WebArena/τ-bench）；单步高≠长程高（误差按步指数衰减）；可靠性工程（错误累积/沙箱/human-in-loop）；MCP | 产业押注最重方向 |
| **世界模型与视频生成** `[全]/[研]` | Sora（DiT 时空 patch）/Veo/Kling；**世界模型=动作条件可交互模拟器（vs 仅好看视频）**；Genie/Genie 2；物理/因果先验假说（尚有争议）；现状校准（物理违和/长时漂移） | 最具想象力也最易炒作 |
| **高效化：蒸馏/量化/小模型** `[全]` | **推理蒸馏（大模型 CoT 数据 SFT 小模型）**；量化（INT4 几乎无损）；**Phi/Gemma/Qwen 小尺寸**；成本/延迟/质量三角；小而专 vs 大而全 | 前沿不只更大，更小更省同样是前沿 |
| **★ 模型版图：判断框架（非型号清单）** `[全]` | 闭源（OpenAI/Anthropic/Google）vs 开放权重（Llama/Qwen/DeepSeek/Mistral/Gemma/Phi）；趋势（开闭差距收窄/推理标配/全面 MoE/降价/中国实验室主力）；**开放权重≠开源+许可陷阱**；**用 LMArena/Artificial Analysis/垂直 benchmark 交叉校准** | **吸收评审：改造为"如何自行评估当下版图"，而非写死型号** |
| **★ 前沿追踪方法论** `[全]` | 信息源分层（arXiv>可信博客>社交噪音）；每周 2–3h 读论文+复现；**三遍法+批判清单**（提升来自方法还是算力/数据？可复现？局限？）；区分真突破/刷榜/营销；最小复现（小模型核心机制）；评测素养（污染/作弊） | **本模块的真正产出：可持续自我更新的肌肉** |

> **吸收评审（时效性 + 评测）**：推理模型确保涵盖 GRPO/RLVR/PRM；缩放纳入推理时缩放；MoE 纳入无辅助损失均衡；位置编码纳入 YaRN/位置插值/attention sink；版图改成判断框架；推理优化跟进 FA-3/prefill-decode 分离/MLA；评测素养与贯穿支线 M-E 联动。

### 关键资源
- **DeepSeek-R1 (arXiv:2501.12948)** (paper) — 最重要精读，可复现蓝本。
- **DeepSeek-V3 (arXiv:2412.19437)** (paper) — 前沿 MoE 标准参考。
- **OpenAI《Learning to Reason with LLMs》(o1 博客) + system cards** (blog) — 区分声称 vs 推断。
- **Lilian Weng《Why We Think》** (blog) — 推理/test-time compute 综述。
- **CS336** (course)。
- **Raschka《Build a LLM》+ ahead-of-AI 博客** (book)。
- **Mixtral (2401.04088) + Switch Transformers (2101.03961)** (paper)。
- **RULER (2404.06654) + Lost in the Middle (2307.03172)** (paper)。
- **SWE-bench/SWE-bench Verified + GAIA/OSWorld/τ-bench 排行榜** (repo)。
- **Sora 技术报告 + DeepMind Genie/Genie 2** (paper)。
- **HF Open-R1 + TRL GRPOTrainer** (repo) — 动手主力。
- **LMArena + Artificial Analysis + Stanford AI Index** (blog) — 交叉校准（引用带日期）。

### 动手项目
1. **核心·复现推理 RL**（TRL GRPOTrainer + 1.5B 基座，GSM8K 子集 RLVR：答案正确+格式奖励，观察 CoT 变长/准确率上升/自我验证，复述 R1-Zero 现象）。
2. **MoE 从零**（nanoGPT 某 block FFN 换最小 MoE，router+Top-2+负载均衡损失，统计专家负载，故意去掉均衡观察坍塌）。
3. 推理蒸馏（用 R1-Distill 或上一步模型生成长 CoT 微调不会推理的同尺寸基座，验证提升）。
4. 长上下文压力测试（needle-in-a-haystack + RULER 风格多针/多跳，画有效准确率 vs 位置/长度曲线，看 lost-in-the-middle）。
5. agent 真实评测（最小 code-agent 在 SWE-bench Lite 跑，记录成功率/失败模式/token 成本 + 反思步骤观察误差累积）。
6. 版图与评测素养（3 个任务各挑开源+闭源模型盲测打分，与 LMArena/Artificial Analysis 对照，产出带日期与方法说明的"模型选型表"，每季度更新）。

### 常见误区
把 test-time compute 当万灵药（简单题过度思考）；误以为 o1/o3 训练细节公开（区分声称/推断/R1 验证）；混淆 RLHF 与 RLVR/推理 RL（奖励来源不同）；GRPO/PPO 栽在工程细节（奖励设计/KL/采样）；迷信长上下文宣称长度（needle≠长程推理）；MoE 参数数字误导（成本看激活参数）；用 demo/单榜判断能力；把"视频好看"等同"懂物理"；追新闻不追机制；忽视开放权重≠开源与许可陷阱。

### 时间估计
作为有限学习单元 5–7 周（核心精读 + 2–3 动手项目），此后转入长期持续追踪（每周 2–3h，无终点）。先用 2 周吃透 test-time compute+R1+MoE 三主干，再 2–3 周覆盖长上下文/多模态/agent/世界模型/高效化，最后 1 周搭版图表与追踪工作流。

### 掌握自检
能白板画训练时 vs 测试时缩放两曲线+解释三形态+举长 CoT 有害场景；能复述 R1 完整流水线+RLVR/GRPO 各解决什么+GRPO 去 critic 好处；能从零实现含 router+Top-k+负载均衡的 MoE+按激活参数估成本；能解释 RoPE 外推/Ring Attention/MLA+用 needle+RULER 证明有效<宣称+说清长上下文 vs RAG；能区分三代多模态+解释 GPT-4o 原生图像为何"理解+生成统一";能用 ≥4 真实基准说明 agent 现状+解释单步高长程低的数学原因；能区分视频生成 vs 可交互世界模型+指出"懂物理"是愿景还是事实；能基于三角论证微调小模型+RAG/工具何时优于超大闭源；能画 2026 年中版图判断框架+说明用哪些榜单交叉校准；能用三遍法+批判清单读最新论文+设计最小复现实验。

---

# 第四部分 · 贯穿式支线（吸收评审补全的横向能力）

> 以下四块横向能力评审指出在原架构中散落或缺失，本课程把它们做成贯穿支线，在多个模块嵌入并在此统一框架。

## 支线 M-E · 评测与方法论（贯穿 M4/M11/M13/M14）
**为什么单列**：对一门"能读懂并复现前沿论文"的课程，评测是核心方法论。
- **经典指标**（M4）：准确率/precision/recall/F1/ROC-AUC/RMSE/MAE。
- **困惑度与内在指标**（M7/M11）：perplexity 的定义、跨 tokenizer 不可比。
- **LLM 基准与污染**（M11/M14）：MMLU/HellaSwag/GSM8K/HumanEval；数据污染 contamination、去污染、基准饱和。
- **LLM-as-judge / 胜率 / Arena**（M12/M14）：AlpacaEval/MT-Bench、长度偏置、LMArena Elo、人类盲评方法学。
- **推理模型评测**（M14）：AIME/GPQA/SWE-bench Verified/ARC-AGI、test-time 投入与方法功劳的区分。
- **RAG/Agent 端到端评估**（M13/M14）：RAGAS（faithfulness/context precision-recall）分段评估、GAIA/OSWorld/τ-bench、单步 vs 长程成功率、pass@k。
- **统一品味**：私有/动态/人类盲评类评测优先于易被刷分的公开静态榜单。

## 支线 M-S · 伦理、安全与负责任 AI（贯穿 M12，独立成框架）
**为什么升级**：面向 2026 这是必备而非可选。
- **对齐相关**（M12）：幻觉、越狱 jailbreak、prompt injection、谄媚 sycophancy、红队 red-teaming。
- **偏见与公平 fairness**：数据去偏、群体公平性指标。
- **隐私与数据治理**：PII 移除、训练数据合规、记忆与抽取攻击。
- **版权与许可**：训练数据版权、开放权重≠开源、商用许可（Apache/MIT vs 社区许可）。
- **AI 治理与监管**：EU AI Act、模型卡 model card、双用途/滥用风险。
- **可扩展监督**（M14）：弱到强泛化、欺骗性对齐 deceptive alignment 等开放问题。

## 支线 M-I · 可解释性与可视化（贯穿 M6/M8/M14）
**为什么补**：2024-26 研究热点，与"读懂前沿"目标直接相关。
- **CNN 可视化**（M6）：卷积核、特征图。
- **经典 ML 可解释性**（M4）：SHAP、LIME、部分依赖图。
- **LLM 可解释性**（M14）：attention 分析、探针 probing、机制可解释性 mechanistic interpretability、**稀疏自编码器 SAE 分解残差流为单义特征**（呼应 M8 稀疏 AE）。

## 概览补充（评审指出的应用域/工程闭环，至少概览级）
- **GNN / 推荐系统 / 表格深度学习**（M4 概览）：工业主力，契合"归纳偏置随数据结构变化"哲学。GNN（消息传递/图卷积）、推荐（双塔召回/排序/向量检索）。
- **经典 NLP 预备**（M7 前最小铺垫）：n-gram 语言模型、文本表示、NER/POS、perplexity。
- **生产/部署 MLOps 闭环**（M13 概览）：服务监控、A/B 测试、漂移检测、回归测试、CI/CD for models、LLM 可观测性（tracing/guardrails）。
- **经典 RL 基础**（M2 速成）：MDP/策略梯度/actor-critic/GAE/PPO（已并入 M2）。

---

# 第五部分 · 贯穿全程的里程碑项目（Capstones）

| 阶段 | 里程碑 | 内容 | 验证 |
|---|---|---|---|
| **阶段 0** | 手写自动微分引擎（micrograd） | 纯 NumPy 矩阵求导链式法则；实现 micrograd 标量自动求导引擎并训玩具任务 | 数学+工程地基扎实（**与阶段1 差异化**） |
| **阶段 1** | 经典 ML 全家桶 + 深网络训练诊断 | (1) 从零线性/逻辑回归/决策树/k-means 对比 sklearn；(2) 纯 NumPy 两层网+MNIST>97%；(3) PyTorch 多层网络 CIFAR-10 + 训练技巧消融报告（初始化/BN/残差/优化器/正则化） | 偏差-方差诊断与训练技巧掌握 |
| **阶段 2** | CNN 图像分类 + RNN 字符级语言模型 | (1) mini-ResNet CIFAR-10>90% + 可视化 + 复现退化；(2) 字符级 LSTM 语言模型 + Bahdanau attention（作为 Transformer 伏笔，**明确这是定性预告而非要求实现 Transformer**） | 归纳偏置直觉 + RNN 瓶颈论证 |
| **阶段 3（核心分水岭）** | 从零实现 nanoGPT | 完全从零 PyTorch 实现多头自注意力/位置编码/Transformer block/decoder-only GPT，tiny-shakespeare 训练采样，逐行解释。配套：VAE/GAN 在 MNIST 生成 | 彻底吃透 Transformer |
| **阶段 4** | 从零 DDPM + 迷你 scaling 实验 | (1) 从零 DDPM 在 MNIST/CIFAR-10 采样；(2) 迷你预训练：训自己的 nanoGPT 验证小型 scaling 趋势（模型/数据规模 vs loss），写观察报告 | 扩散数学 + 缩放认知 |
| **阶段 5（终极）** | 端到端 LLM 应用 + 对齐复现 | (1) LoRA/QLoRA 微调开源模型完成特定任务（产出 SFT 数据+评测）；(2) 带检索的 RAG/Agent 应用（向量库+检索+重排+工具调用，RAGAS 诊断改进）；(3) 复现 DPO 并对比 SFT 基线胜率。整合成作品集仓库 + vLLM 部署 | 端到端能力闭环 |
| **阶段 5 进阶** | 复现推理 RL（研究路线） | 用 GRPO+RLVR 在 GSM8K 子集让小模型学会推理，复述 R1-Zero 现象 | 把 RL 打通到前沿 |
| **贯穿全程（研究路线）** | 论文复现挑战 | 从阶段3 起每阶段复现 1 篇经典论文的核心结果或关键消融（attention 变体/RLHF-DPO 对比/2024-26 推理模型核心机制） | 独立研究与复现能力 |

---

# 第六部分 · 推荐总资源清单

## 课程（一手优先）
- **3Blue1Brown**（线代/微积分/神经网络可视化）
- **CS231n**（视觉/CNN/反向传播）
- **CS224n**（NLP/序列/Transformer）
- **CS229**（经典 ML 严谨推导）
- **CS336: Language Modeling from Scratch**（从零造 LLM，对接阶段4/5）
- **MIT 18.06**（线代）/ **MIT 6.431x**（概率）/ **EE364A**（凸优化）
- **Hugging Face NLP Course + Deep RL Course**
- **Andrew Ng ML Specialization**
- **OpenAI Spinning Up**（RL/PPO）

## 书籍（多数免费在线）
- 《Mathematics for Machine Learning》、《Deep Learning》（花书）、PRML/《Deep Learning: Foundations and Concepts》(Bishop)
- 《An Introduction to Statistical Learning》(ISLP)、《Hands-On ML》(Géron)
- 《Deep Learning with PyTorch》、Dive into Deep Learning (D2L)
- 《Neural Networks and Deep Learning》(Nielsen)、Sutton & Barto《Reinforcement Learning》
- 《Build a LLM from Scratch》(Raschka)、《Interpretable ML》(Molnar)

## 关键论文（按主线）
- **基础架构**：Attention Is All You Need、ResNet、LSTM、Bahdanau attention、ViT、RoFormer(RoPE)
- **生成模型**：VAE、GAN、DDPM、DDIM、Score-based SDE、Latent Diffusion、CFG、ControlNet
- **表示/多模态**：word2vec、GloVe、SimCLR、CLIP、LLaVA、BLIP-2
- **预训练/缩放**：GPT-3、Kaplan/Chinchilla、Llama 技术报告、Switch/Mixtral、DeepSeek-V3
- **对齐**：InstructGPT、DPO、Constitutional AI、KTO/ORPO/IPO、Llama 2
- **应用/推理**：LoRA、QLoRA、RAG、Lost in the Middle、ReAct、FlashAttention、PagedAttention
- **前沿**：DeepSeek-R1、Sora、Genie、RULER

## 博客/工具/仓库
- **Karpathy 全系列**：micrograd → makemore → minGPT/nanoGPT → minbpe → build-nanogpt（implement-to-understand 脊柱）
- **Lilian Weng（lil'log）**：扩散、对比学习、prompt、agent、推理综述
- **Jay Alammar**：Illustrated Transformer/GPT-2/Word2vec
- **The Annotated Transformer / Diffusion Model**（逐行实现）
- **distill.pub / Seeing Theory / CNN Explainer / bbycroft.net/llm**（交互可视化）
- **工具栈**：PyTorch、Hugging Face (transformers/datasets/peft/trl/diffusers/accelerate)、vLLM、W&B/TensorBoard、FAISS/Chroma、RAGAS、lm-evaluation-harness、Open-R1

---

# 第七部分 · 学完之后如何持续跟踪前沿

> M14 没有终点——本课程的真正产出不是记住 2026 年某个排名，而是带走一套"读论文—复现—质疑"的肌肉，让你在 2027、2028 也能自己跟上。

## 1. 建立固定节律
- 每周固定 **2–3 小时** 读论文 + 小复现，可持续性远胜单次冲刺。
- 维护一个第二大脑（Notion/Obsidian/Anki），记录推导、踩坑、论文要点，做间隔重复。

## 2. 信息源分层
| 层级 | 来源 |
|---|---|
| 一手（优先） | arXiv（cs.CL/cs.LG）、官方技术报告/system card、模型权重页 |
| 高质量二手 | 少数可信研究者博客/newsletter（Lilian Weng、Sebastian Raschka、Sebastian Ruder）、会议 tutorial |
| 噪音（警惕） | 社交媒体热搜、营销稿、未经独立复现的"突破" |

## 3. 读论文"三遍法 + 批判清单"
1. 先读摘要+图表抓骨架 → 2. 读方法抓核心 → 3. 读细节能复现。
- **批判清单**：它解决什么问题？基线是否公平？提升来自方法还是更多算力/数据？是否可复现（有无代码/权重）？失败模式与未讨论的局限是什么？

## 4. 区分"真突破/刷榜/营销"的启发式
- 是否有独立复现、是否在多个独立 benchmark 一致、是否报告成本与失败案例、是否混淆 test-time compute 投入与方法本身的功劳。

## 5. 最小复现策略
- 不追求复现全规模，在小模型/小数据上复现"核心机制"（如用 GRPO 在 GSM8K 子集让 0.5B–1.5B 模型学会推理；用 nanoGPT 加一个 MoE 层观察路由）。

## 6. 评测素养
- 知道每个 benchmark 测什么、易被污染/作弊在哪，优先信任私有/动态/人类盲评类评测。

## 7. 持续校准模型版图
- 用 **LMArena（人类盲评 Elo）+ Artificial Analysis（成本/速度/质量）+ 垂直 benchmark（SWE-bench/GPQA/AIME/RULER）** 交叉看，不信单一榜单或厂商自报数字。每季度更新一次你自己的"模型选型表"（带日期与方法说明）。

## 8. 选定方向深耕（研究路线）
- 在对齐、长上下文、推理、多模态、可解释性、效率中选 1–2 个细分方向，持续产出复现与改进，形成自己的研究品味。

> **终点不是学完，而是能就某一细分方向持续产出复现与改进。带着这套能力，你已经从"会调 API"升级为"能读懂、复现、评判并推进前沿"的人。**