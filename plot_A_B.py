import pandas as pd
import matplotlib.pyplot as plt

# Đổi tên file nếu bạn lưu khác
csv_path = "schedule_trace.csv"

df = pd.read_csv(csv_path)
assert {"t", "A", "B"}.issubset(df.columns), "CSV cần có cột: t, A, B"

# (Tuỳ chọn) nền tối
# plt.style.use("dark_background")

plt.figure()
plt.plot(df["t"], df["A"], label="A(t)", linewidth=2)
plt.plot(df["t"], df["B"], label="B(t)", linewidth=2, linestyle="--")
plt.xlabel("t")
plt.ylabel("A(t), B(t)")
plt.title("Anneal schedules A(t), B(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("schedule_At_Bt.png", dpi=200)
plt.show()
