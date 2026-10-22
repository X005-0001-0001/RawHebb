# -*- coding: utf-8 -*-
"""Build the 0.9.265 branch and 0.20.0.0 release (312-faithful shell)."""
import io

SRC = r'D:\explore\wav\RawHebb+shell19+0.9.264.0.py'
BRANCH = r'D:\explore\wav\RawHebb+shell20+0.9.265.0.py'
RELEASE = r'D:\explore\wav\RawHebb+bio+0.20.0.0.py'

t = io.open(SRC, encoding='utf-8').read()

# header block
start = t.index('0.19.0.0 做了什么')
end = t.index('词表\n    基座', start)
NEW = '''0.20.0.0 做了什么（把 3120012 的壳子拔下来，适配 RawHebb 内核）
    · 渲染层直接移植 3120012：<|bos|> [tools] system/user/tool/assistant 段 <|eos|>；
      assistant 的 reasoning_content + content + tool_calls 参与训练，其余 -100。
    · 上下文截断直接移植 `_truncate_pairs`：保留开头 system + 最近完整问答对，
      预算用真实 token 数（含 role/end/reasoning/tool_calls），不在句子中间硬切。
    · 数据管线：.txt = 预训练（整文件窗口、全部 token、默认不验证、窗口按 seed 打乱）；
      .jsonl = 对话（assistant mask；验证集从同一文件按**记录**随机抽 10%，或显式 val）。
    · 验证与训练使用同一 SEQ_LEN 窗口（312 口径）；旧的“最后 128 token”不再用于训练内验证。
      因此 0.20 的 CE 与 0.15–0.19 的 128-token 数字不可直接比较。
    · 词表完整生命周期：训练 / 保存 / 加载 / 元数据校验 / 基座回退；重建词表后旧权重明确拒绝。
    · 在线学习（菜单 13）：文件增量逐条更新 / 对话中学习；必须从已训练权重开始。
    · 菜单恢复 312 的选文件、容错输入、确认与返回；配置删死参数并注明来源。

模型
    · 配方沿用 0.18：hidden 432 × 4 层、词向量冻结、SEQ_LEN=384、1M int16 trigram 桶、
      4k last-token 计数、行归一化 bag 先验（Jeffreys 0.5）；纯 CPU 局部 Hebbian。
    · 本轮不重新训练、不调参；只修外壳与数据口径。

'''
t = t[:start] + NEW + t[end:]
t = t.replace('训练窗口 SEQ_LEN=384（完整对话，语料均值 472 token）；评测只取最后 EVAL_LEN=128 token。',
              '训练与验证同一窗口 SEQ_LEN=384；验证集按记录抽样，不再另取 128 token。')
t = t.replace('1  训练模型（可选择 .jsonl，默认 train_dsr1.jsonl）',
              '1  训练模型（.jsonl 对话 / .txt 预训练）')
t = t.replace('验证集默认从同一文件按**对话**随机切 10%（seed 固定），也可显式给 val 文件；',
              '验证集默认从同一文件按**记录**随机抽 10%（seed 固定），也可显式给 val 文件；')
t = t.replace('超过 SEQ_LEN 时保留 system + 最近完整问答对，不在句子中间硬切。',
              '超过 SEQ_LEN 时按 312 的 `_truncate_pairs` 保留 system + 最近完整问答对。')
t = t.replace('RawHebb+shell19+0.9.264.0', 'RawHebb+shell20+0.9.265.0')
t = t.replace("APP_BRANCH = 'shell19'", "APP_BRANCH = 'shell20'", 1)
t = t.replace("APP_VERSION = '0.9.264.0'", "APP_VERSION = '0.9.265.0'", 1)
t = t.replace('#   0.19.0.0 2026-10-22  外壳工程版',
              '#   0.20.0.0 2026-10-22  照 3120012 重写数据壳：渲染/mask/问答对截断/按记录切分/\n'
              '#                         验证同窗；修 0.19 的菜单 NameError；在线学习需已训练权重。\n'
              '#   0.19.0.0 2026-10-22  外壳工程版', 1)
io.open(BRANCH, 'w', encoding='utf-8', newline='\n').write(t)

r = t.replace('RawHebb+shell20+0.9.265.0', 'RawHebb+bio+0.20.0.0')
r = r.replace("APP_BRANCH = 'shell20'", "APP_BRANCH = 'bio'", 1)
r = r.replace("APP_VERSION = '0.9.265.0'", "APP_VERSION = '0.20.0.0'", 1)
io.open(RELEASE, 'w', encoding='utf-8', newline='\n').write(r)
print('wrote', BRANCH)
print('wrote', RELEASE)
