#!/usr/bin/env python3
"""
生成第2章 图2-5：链式法则计算图（优化版v3 - 减少空白）
使用 Matplotlib，紧凑布局
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── 中文字体设置 ───
plt.rcParams['font.family'] = 'PingFang HK'
plt.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════
# 数值
# ════════════════════════════════════════════════════
x_val, w_val, b_val = 2.0, 0.5, 0.1
u_val = w_val * x_val + b_val            # 1.1
y_val = 1 / (1 + np.exp(-u_val))         # 0.7503
L_val = 0.5 * (y_val - 1.0)**2           # 0.0312

dL_dL = 1.0
dL_dy = y_val - 1.0                      # -0.2497
dy_du = y_val * (1 - y_val)              # 0.1874
dL_du = dL_dy * dy_du                    # -0.0468
dL_dx = dL_du * w_val                    # -0.0234
dL_dw = dL_du * x_val                    # -0.0936
dL_db = dL_du * 1.0                      # -0.0468

# ════════════════════════════════════════════════════
# 画图 — 紧凑布局
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-2.0, 4.5)
ax.set_aspect('equal')
ax.axis('off')

# ════════════════════════════════════════════════════
# 颜色
# ════════════════════════════════════════════════════
C_INPUT  = "#d5e8d4"
C_NODE   = "#fff8e1"
C_LOSS   = "#fce4ec"
C_FWD    = "#2e7d32"
C_BWD    = "#d32f2f"
C_GRADBG = "#ffebee"
C_FWDBG  = "#e8f5e9"

# ════════════════════════════════════════════════════
# 节点位置 — 更紧凑
# ════════════════════════════════════════════════════
NX, NW, NB = 0.2, 0.2, 0.2
nodes = {}
nodes["x"] = (NX, 2.8, 1.0, 0.7)
nodes["w"] = (NW, 1.0, 1.0, 0.7)
nodes["b"] = (NB, -0.8, 1.0, 0.7)
nodes["linear"]  = (4.5, 0.8, 2.2, 1.5)
nodes["sigmoid"] = (7.8, 0.8, 2.2, 1.5)
nodes["loss"]    = (11.0, 0.8, 2.2, 1.5)

def draw_rect(x, y, w, h, color, title, formula="", fwd="", bwd=""):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="#666", linewidth=1.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.25, title,
            ha='center', va='center', fontsize=10, fontweight='bold')
    if formula:
        ax.text(x+w/2, y+h/2+0.05, formula,
                ha='center', va='center', fontsize=8.5, color="#666")
    if fwd:
        ax.text(x+w/2, y+0.20, f"前向: {fwd}",
                ha='center', va='center', fontsize=8,
                color=C_FWD, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=C_FWDBG, edgecolor='none'))
    if bwd:
        ax.text(x+w/2, y+0.06, f"梯度: {bwd}",
                ha='center', va='center', fontsize=8,
                color=C_BWD, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=C_GRADBG, edgecolor='none'))

# ── 绘制节点 ──
draw_rect(*nodes["x"], C_INPUT, "x（输入）", "", f"x = {x_val}")
draw_rect(*nodes["w"], C_INPUT, "w（权重）", "", f"w = {w_val}")
draw_rect(*nodes["b"], C_INPUT, "b（偏置）", "", f"b = {b_val}")
draw_rect(*nodes["linear"], C_NODE, "线性变换",
          "u = w·x + b", f"u = {u_val:.3f}", f"dL/du = {dL_du:.3f}")
draw_rect(*nodes["sigmoid"], C_NODE, "Sigmoid 激活",
          "y = σ(u)", f"y = {y_val:.3f}", f"dL/dy = {dL_dy:.3f}")
draw_rect(*nodes["loss"], C_LOSS, "损失函数",
          "L = ½(y-t)²", f"L = {L_val:.4f}", f"dL/dL = {dL_dL:.1f}")

# ════════════════════════════════════════════════════
# 前向箭头（绿色实线）
# ════════════════════════════════════════════════════
def arrow_fwd(x1, y1, x2, y2, label="", ox=0, oy=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_FWD, lw=2.5),
                zorder=3)
    if label:
        mx, my = (x1+x2)/2+ox, (y1+y2)/2+oy
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=7.5, color=C_FWD, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=C_FWDBG, edgecolor='none', alpha=0.9))

arrow_fwd(nodes["x"][0]+nodes["x"][2], nodes["x"][1]+nodes["x"][3]/2,
          nodes["linear"][0], nodes["linear"][1]+nodes["linear"][3]*0.75, "x")
arrow_fwd(nodes["w"][0]+nodes["w"][2], nodes["w"][1]+nodes["w"][3]/2,
          nodes["linear"][0], nodes["linear"][1]+nodes["linear"][3]*0.5, "w")
arrow_fwd(nodes["b"][0]+nodes["b"][2], nodes["b"][1]+nodes["b"][3]/2,
          nodes["linear"][0], nodes["linear"][1]+nodes["linear"][3]*0.25, "b")
arrow_fwd(nodes["linear"][0]+nodes["linear"][2], nodes["linear"][1]+nodes["linear"][3]/2,
          nodes["sigmoid"][0], nodes["sigmoid"][1]+nodes["sigmoid"][3]*0.65, "u")
arrow_fwd(nodes["sigmoid"][0]+nodes["sigmoid"][2], nodes["sigmoid"][1]+nodes["sigmoid"][3]/2,
          nodes["loss"][0], nodes["loss"][1]+nodes["loss"][3]*0.65, "y")

# ════════════════════════════════════════════════════
# 反向箭头（红色虚线）
# ════════════════════════════════════════════════════
def arrow_bwd(x1, y1, x2, y2, label="", ox=0, oy=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_BWD, lw=2.0, linestyle='dashed'),
                zorder=3)
    if label:
        mx, my = (x1+x2)/2+ox, (y1+y2)/2+oy
        ax.text(mx, my, label, ha='center', va='top',
                fontsize=7.5, color=C_BWD, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=C_GRADBG, edgecolor='#ef9a9a', linewidth=0.5))

arrow_bwd(nodes["loss"][0], nodes["loss"][1]+nodes["loss"][3]*0.35,
          nodes["sigmoid"][0]+nodes["sigmoid"][2], nodes["sigmoid"][1]+nodes["sigmoid"][3]*0.35,
          "dL/dL=1", ox=-1.3)
arrow_bwd(nodes["sigmoid"][0], nodes["sigmoid"][1]+nodes["sigmoid"][3]*0.35,
          nodes["linear"][0]+nodes["linear"][2], nodes["linear"][1]+nodes["linear"][3]*0.35,
          "dL/dy=-0.250", ox=-1.5)
arrow_bwd(nodes["linear"][0], nodes["linear"][1]+nodes["linear"][3]*0.25,
          nodes["x"][0]+nodes["x"][2], nodes["x"][1]+nodes["x"][3]/2,
          "dL/du=-0.047")
arrow_bwd(nodes["linear"][0], nodes["linear"][1]+nodes["linear"][3]*0.5,
          nodes["w"][0]+nodes["w"][2], nodes["w"][1]+nodes["w"][3]/2,
          "dL/dw=-0.094")
arrow_bwd(nodes["linear"][0], nodes["linear"][1]+nodes["linear"][3]*0.75,
          nodes["b"][0]+nodes["b"][2], nodes["b"][1]+nodes["b"][3]/2,
          "dL/db=-0.047")

# ════════════════════════════════════════════════════
# 图例
# ════════════════════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor=C_FWDBG, edgecolor=C_FWD, label='前向传播'),
    mpatches.Patch(facecolor=C_GRADBG, edgecolor=C_BWD, label='反向传播'),
]
legend = ax.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.35), ncol=2,
                   framealpha=0.9, fontsize=9,
                   markerscale=0.8)
legend.get_frame().set_linewidth(0.5)

# ════════════════════════════════════════════════════
# 保存 — tight_layout 去掉多余空白
# ════════════════════════════════════════════════════
output_path = "《AI神经网络的数学——用Python、PyTorch讲解》/images/ch02/NN02_chain_rule_graph.png"
plt.tight_layout(pad=0.3)
plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
plt.close()
print(f"✅ 已生成：{output_path}")
print(f"   figsize=12x4.5, dpi=200")
