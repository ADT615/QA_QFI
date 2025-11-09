import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("qfi_trace.csv")   # cần cột: t,FQ (t là thời gian thực)
t  = df["t"].to_numpy()
FQ = df["FQ"].to_numpy()

# Tránh chia cho 0 ở t=0: bỏ điểm đầu hoặc dùng epsilon nhỏ
eps = 1e-12
y = FQ / t**2

plt.figure()
plt.plot(t, y, lw=2)
plt.xlabel("t")
plt.ylabel(r"$F_Q(t) / t^2$")
#plt.title(r"Scaled QFI: $F_Q(t)/t^2$")
plt.title("Quantum Fisher Information vs. Time")
# (tuỳ chọn) hộp thông số
N, J, h, K, alpha, T = 4, 1.0, 0.2, 2000, 1.0, 10  # sửa cho đúng run của bạn
info = [
    f"N = {N}", f"J = {J}", f"h = {h}",
    f"T = {T}", f"K = {K}", rf"$\alpha$ = {alpha}",
    rf"A(t)=1-s(t)", rf"B(t)=s(t)", rf" $s(t)=(t/T)^{{\alpha}}$"
]
ax = plt.gca()
ax.text(0.02, 0.98, "\n".join(info), transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.15))

plt.grid(True)
plt.tight_layout()
plt.savefig("FQ_div_t2.png", dpi=200)
plt.show()
