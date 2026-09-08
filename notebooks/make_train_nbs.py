#!/usr/bin/env python3
"""Generate one self-contained training notebook per experiment.

Why not one notebook with a config switch
-----------------------------------------
E1_colab.ipynb sets the experiment in Cell 1 and every other cell reads it. That is
tidy until two experiments have to run at once: the two runs then live in two browser
tabs holding two different values of the same variable, and any edit to the shared
notebook forces a reload -- which drops whichever session was mid-training. That is
exactly what happened on 2026-09-08.

So each experiment gets its own notebook with its own values already filled in. Nothing
to edit before running, nothing shared to invalidate, and two tabs cannot be confused
for each other because the title and the run name are printed by the first cell.

There is also no config assertion here. E1_colab's check_config() existed because its
cells could be pointed at the wrong config; these cannot. The distinguishing settings
are printed instead, which is what actually answers "which architecture is training" --
Colab truncates the resolved-config dump at the top of a run.

    python notebooks/make_train_nbs.py
"""
import io
import json


def lines(text):
    """Split for the .ipynb `source` field, which keeps the newlines.

    A plain split drops them and Jupyter joins the list into one line, so every
    statement runs together and a leading `!` comes back as a bash syntax error.
    """
    ls = text.strip('\n').split('\n')
    return [l + '\n' for l in ls[:-1]] + [ls[-1]]


def build(path, title, intro, config, run_name, epochs, needs_masks):
    cells = []

    def md(t):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': lines(t)})

    def code(t):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': lines(t)})

    md("""
# %s

%s

**这个 notebook 的参数已经填好，不需要修改任何一行。** 运行时选 A100。

顺序：`1 → 2 → 3 → 4（冒烟）→ 5（短程 20 epoch）→ 6（全量）`。
掉线后：`1 → 2 → 3 → 7（续跑）`。

`run name` 是 `%s`，checkpoint 与日志都写在 Drive 的这个名字下面，
所以本 notebook 与另一个训练 notebook 可以同时运行，互不干扰。
""" % (title, intro, run_name))

    md('## 1. 挂载 Drive、设定参数')

    code("""
from google.colab import drive
import os, yaml

drive.mount('/content/drive')

os.environ['DRIVE']    = '/content/drive/MyDrive/SA-CUT'
os.environ['REPO_URL'] = 'https://github.com/z-pan/SA-CUT.git'
os.environ['BRANCH']   = 'main'
os.environ['CONFIG']   = '%s'
os.environ['RUN_NAME'] = '%s'

assert os.path.isdir(os.environ['DRIVE']), 'Drive 目录没找到: ' + os.environ['DRIVE']
print('run     :', os.environ['RUN_NAME'])
print('config  :', os.environ['CONFIG'])
print('GPU     :', end=' ')
os.system('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
""" % (config, run_name))

    md("""
**确认 GPU 是 A100。** 两个 notebook 同时跑时，Colab 可能只给其中一个 GPU，
另一个退化到 CPU 会慢到不可用——在这里就要看出来，不要等训练开始。
""")

    md('## 2. 取代码')

    code("""
%cd /content
!if [ -d SA-CUT ]; then cd SA-CUT && git fetch -q && git checkout -q "$BRANCH" && git pull -q --ff-only; else git clone -q --branch "$BRANCH" "$REPO_URL"; fi
%cd /content/SA-CUT
!pip install -q tifffile pytorch-fid pyyaml scikit-image 2>&1 | tail -2
!git log --oneline -1
""")

    md("""
## 3. 数据 Drive → 本地磁盘

`--ignore-existing` 让重连后这一格几乎瞬间完成。%s
""" % ('CUT 不使用核 mask（`mask_provider.mode: none`），mask 一并同步不影响。'
       if not needs_masks else 'mask 必须与 TPAF patch 按文件名一一对应。'))

    code("""
!mkdir -p data/raw/tpaf data/raw/he data/patches/masks
!rsync -a --ignore-existing "$DRIVE/patches/tpaf/"  data/raw/tpaf/
!rsync -a --ignore-existing "$DRIVE/patches/he/"    data/raw/he/
!rsync -a --ignore-existing "$DRIVE/patches/masks/" data/patches/masks/
!echo "tpaf=$(ls data/raw/tpaf | wc -l)  he=$(ls data/raw/he | wc -l)  masks=$(ls data/patches/masks | wc -l)"
""")

    md("""
## 4. 冒烟测试（约 30 秒）

CPU、1 epoch、极小的合成数据。验的是环境和这份配置能不能跑通，不是训练。
同时把这个实验区别于其它实验的字段打印出来——Colab 会截断开头的完整配置输出，
除此之外整个日志里没有地方说明在训哪个架构。
""")

    code("""
cfg = yaml.safe_load(open(os.environ['CONFIG'], encoding='utf-8'))
def dig(c, path):
    for k in path.split('.'):
        c = c.get(k) if isinstance(c, dict) else None
    return c
for f in ('model.type', 'generator.input_nc', 'generator.mask_injection',
          'generator.decoder_upsample', 'mask_provider.mode',
          'losses.lambda_struct', 'training.lr_D',
          'training.d_loss_gate_threshold'):
    print('  %-34s %s' % (f, dig(cfg, f)))

!bash scripts/smoke_test.sh --config "$CONFIG"
""")

    md("""
## 5. 短程验证 —— 20 epoch，先跑这一格

**不要跳过。** 约全量的二十分之一，用来在投入几小时 A100 之前确认对抗博弈是健康的。
写到 `%s_short`，不会覆盖正式 run 的 checkpoint。

日志里要看的：

| 字段 | 健康 | 有问题 |
|---|---|---|
| `D=` | 接近 **0.25** | `< 0.15` → 判别器赢了，生成器只是在给 TPAF 上色 |
| `D= (real … / fake …)` | 两者明显分开 | 都漂到 0.25 附近 → 判别器没有信息量 |
| `G=` | 有起伏但不发散 | 单调上升 → 生成器追不上 |
""" % run_name)

    code("""
!python scripts/train.py \\
    --config "$CONFIG" \\
    --training.n_epochs=20 \\
    --training.n_epochs_decay=0 \\
    --data.patch_size=512 \\
    --data.num_workers=2 \\
    --experiment.name="%s_short" \\
    --experiment.checkpoint_dir="$DRIVE/checkpoints" \\
    --experiment.log_dir="$DRIVE/results/logs" \\
    --experiment.use_wandb=false
""" % run_name)

    md("""
## 6. 全量训练（%d epoch）

`latest.pth` 每个 epoch 结束都写一次 Drive，带编号的每 10 个 epoch 写一次，
所以掉线最多损失一个 epoch。掉线后跑第 7 格续。
""" % epochs)

    code("""
!python scripts/train.py \\
    --config "$CONFIG" \\
    --data.patch_size=512 \\
    --data.num_workers=2 \\
    --experiment.name="$RUN_NAME" \\
    --experiment.checkpoint_dir="$DRIVE/checkpoints" \\
    --experiment.log_dir="$DRIVE/results/logs" \\
    --experiment.use_wandb=false
""")

    md("""
## 7. 掉线后续跑

先跑 1、2、3 三格，再跑这一格。`--resume` 会恢复权重和优化器，
并把学习率调度器快进到对应的 epoch。

`D_RESCUE` 默认为空。只有在日志显示判别器赢了（`d_loss_ema` 掉到 0.3 以下）
时才取消下面那行的注释——E1 就是在 epoch 186 出现这个情况（`d_loss_ema = 0.129`）。
""")

    code("""
D_RESCUE = ''
# D_RESCUE = '--training.d_loss_gate_threshold=0.2 --training.lr_D=5e-5'

!python scripts/train.py \\
    --config "$CONFIG" \\
    --resume "$DRIVE/checkpoints/$RUN_NAME/latest.pth" \\
    {D_RESCUE} \\
    --data.patch_size=512 \\
    --data.num_workers=2 \\
    --experiment.name="$RUN_NAME" \\
    --experiment.checkpoint_dir="$DRIVE/checkpoints" \\
    --experiment.log_dir="$DRIVE/results/logs" \\
    --experiment.use_wandb=false
""")

    md("""
## 8. 看已存的 checkpoint

训练中途随时可以另开一格跑。下载时**多带几个 epoch 的**——
checkpoint 要用训练病例自己的真实 H&E 做分布验证来选，不能用评估区域挑。
""")

    code("""
!ls -lh "$DRIVE/checkpoints/$RUN_NAME/" | tail -20
!tail -5 "$DRIVE/results/logs/$RUN_NAME/train.log"
""")

    nb = {'cells': cells,
          'metadata': {'accelerator': 'GPU',
                       'colab': {'provenance': [], 'gpuType': 'A100'},
                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 0}
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('wrote %s, %d cells' % (path, len(cells)))


build('notebooks/train_sa_cut_v2.ipynb',
      'SA-CUT v2 训练',
      '把 E1 的转置卷积解码器换成 resize-conv（最近邻上采样 + stride-1 卷积）。'
      'E1 的 `ConvTranspose2d(kernel=3, stride=2)` 中 3 不能被 2 整除，'
      '这正是那层编织状棋盘伪影的成因。判别器也重新平衡过'
      '（`lr_D` 5e-5、gate 0.2）。**权重形状变了，必须从零训练，不能从 E1 续。**',
      'configs/experiment_sa_cut_v2.yaml', 'sa_cut_v2', 400, True)

build('notebooks/train_cut_baseline.ipynb',
      'vanilla CUT 训练（对比方法）',
      '标准 CUT（Park et al., ECCV 2020），不做任何结构性改动：'
      '`model.type=cut`、`input_nc=1`（无 mask 通道）、`patchnce_mode=standard`、'
      '`lambda_struct=0`、`lambda_idt=0`、`mask_provider.mode=none`。'
      'SA-CUT 相对它多的正是「mask 输入 + SA-PatchNCE + L_struct」三项，'
      '所以两者之差就是核 mask 在 CUT 骨架上的贡献。',
      'configs/ablation_cut_baseline.yaml', 'E2_cut_baseline', 400, False)
