# -*- coding: utf-8 -*-
"""v119 生成器：在 v118「零魔数」基础上修两处（v118 的实测失败原因见下）。

v118 实测：验证 CE 7.84（≈随机 7.92）、top1 2.15%，混合权重塌成 [0.0, 0.0, 0.0001]。
两个原因都在记忆这一侧：

  ① 记忆写入用了「全程运行均值」lr=1/(n+1) ⇒ 每个列原型是**整个历史**的平均，
     recency 被彻底抹平；而所有列的平均向量都接近全局均值 ⇒ 列原型彼此共线 ⇒
     argmax 命中率≈随机（1/V）。
     修：**指数平均，衰减直接取本层的膜泄漏**（1−leak）。
     这不是新常数——用的就是系统里已有的那套生物时间常数（L0 leak=0.9048，
     深层 0.951/0.975/0.988），语义上也正是「短期记忆有多短」。

  ② 混合权重按命中率给，但**低于随机也算证据**。修：
     `w_k = max(hit_k/seen_k − 1/V, 0)`，再归一化。
     `1/V` 是词表给出的结构量（随机基线），不是手调的超参；语义是
     「还没证明自己比瞎猜强，就闭嘴」。

保留 v118 的：W_out 方向归一 + 1/(t+1)、删除 γ 与所有护栏、问答 mask、多时间常数。
"""
import io
import sys

SRC = 'wavesnn_hebbian_v118.py'
DST = 'wavesnn_hebbian_v119.py'
src = io.open(SRC, encoding='utf-8').read()

R = []

R.append((
    """V118_MIX = os.environ.get('V118_MIX', '1') != '0'            # 混合系数 = 各层运行命中率
V118_MEAN_MEM = os.environ.get('V118_MEAN_MEM', '1') != '0'  # 记忆写入 = 逐列运行均值""",
    """V119_MIX = os.environ.get('V119_MIX', '1') != '0'            # 混合系数 = 各层「超随机」的命中率
V119_EMA_MEM = os.environ.get('V119_EMA_MEM', '1') != '0'    # 记忆写入 = 以膜泄漏为衰减的指数平均""", 1))

R.append((
    """            _hit = self._v118_hit[_v112_idx]
            _tot = self._v118_seen[_v112_idx].clamp(min=1.0)
            w = _hit / _tot""",
    """            _hit = self._v118_hit[_v112_idx]
            _tot = self._v118_seen[_v112_idx].clamp(min=1.0)
            # ★v119：只有**超过随机基线**的证据才算数。1/V 是词表给出的结构量（瞎猜的水平），
            #   不是手调的超参；语义 =「还没证明自己比瞎猜强，就闭嘴」。
            w = (_hit / _tot - 1.0 / self.vocab_size).clamp(min=0.0)""", 1))

R.append((
    """                            if V118_MEAN_MEM:
                                # ★v118：逐列运行均值 W ← (n·W + V)/(n+1)，等价 lr=1/(n+1)
                                #   零常数、凸组合（天然不过冲）⇒ 不需要衰减来救
                                _n = float(self._v118_mcount[_li112, _tid])
                                _cur = self.W_m_vocab[_li112].data[:, _tid]
                                self.W_m_vocab[_li112].data[:, _tid] = (
                                    (_n * _cur + _hT112[0, _t112]) / (_n + 1.0))
                                self._v118_mcount[_li112, _tid] = _n + 1.0""",
    """                            if V119_EMA_MEM:
                                # ★v119：指数平均，**衰减直接取本层的膜泄漏**。
                                #   v118 用全程运行均值（lr=1/(n+1)）把 recency 抹平了，
                                #   列原型彼此共线 ⇒ argmax 命中率≈随机 ⇒ 混合权重塌成 0。
                                #   这里的 (1−leak) 不是新常数：它就是系统里已有的那套
                                #   生物时间常数（L0 0.9048 / 深层 0.951、0.975、0.988）。
                                _lay = self.layers[_li112]
                                _a = float(1.0 - (getattr(_lay, 'leak_v112', None)
                                                 if _li112 >= 1 else _lay.leak))
                                _cur = self.W_m_vocab[_li112].data[:, _tid]
                                self.W_m_vocab[_li112].data[:, _tid] = (
                                    (1.0 - _a) * _cur + _a * _hT112[0, _t112])
                                self._v118_mcount[_li112, _tid] += 1.0""", 1))

for i, (o, n, c) in enumerate(R):
    got = src.count(o)
    if got != c:
        print('★第 %d 处命中 %d（要求 %d）：%r' % (i, got, c, o[:110]))
        sys.exit(1)
    src = src.replace(o, n, 1)

src = src.replace('V118_MIX', 'V119_MIX').replace('V118_MEAN_MEM', 'V119_EMA_MEM')
src = src.replace('wavesnn_hebbian_v118', 'wavesnn_hebbian_v119')
io.open(DST, 'w', encoding='utf-8', newline='\n').write(src)
print('已生成 %s' % DST)