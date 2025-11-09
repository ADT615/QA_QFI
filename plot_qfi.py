import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("qfi_trace.csv")

plt.figure()
plt.plot(df["t"], df["FQ"])
plt.xlabel("t")
plt.ylabel("FQ(t)")
plt.title("Quantum Fisher Information vs. Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("qfi_vs_t.png", dpi=160, bbox_inches='tight')
plt.show()
