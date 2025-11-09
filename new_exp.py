import numpy as np
from scipy.linalg import expm, eigh
from numpy import kron

# ---------- Utilities ----------

def pauli_mats():
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

def op_on_site(single_op, site, N):
    I, _, _, _ = pauli_mats()
    ops = [I]*N
    ops[site] = single_op
    M = ops[0]
    for k in range(1, N):
        M = np.kron(M, ops[k])
    return M

def sum_two_body_ZZ(N, J = 1.0, open_chain = True):
    _, _, _, Z = pauli_mats()
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    last = N-1
    pairs = [(j,j+1) for j in range(N-1)]
    if not open_chain:
        pairs.append((last, 0))
    for (j,k) in pairs:
        H += J* (op_on_site(Z, j , N) @ op_on_site(Z, k , N))
    return H

def sum_one_body_X(N):
    _, X, _, _ = pauli_mats()
    dim = 2**N 
    Xsum = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        Xsum += op_on_site(X, j, N)
    return Xsum

def ground_state(H):
    E, V = eigh(H+H.T.conj()/2)
    return E[0], V[:,0]

# ---------- Annealing schedule ----------

def s_of_t(t, T, alpha=1.0):
    s = (t/T) ** alpha
    return min(max(s, 0.0),1.0)

def A_of_t(t, T, alpha=1.0):
    return 1.0 - s_of_t(t, T, alpha)

def B_of_t(t, T, alpha=1.0):
    return s_of_t(t, T, alpha)

def pick_steps_per_unit(N, J, h, safety=0.1):
    # ước lượng ||H||_max ~ |J|(N-1) + |h| N
    Hscale = abs(J)*(N-1) + abs(h)*N
    dt_target = safety / max(Hscale, 1e-12)   # safety=0.1 hoặc 0.05
    return int(np.ceil(1.0 / dt_target))

# calculate energy

# def energy_spectrum(N, J, h, T, alpha, open_chain = True, num_points, k_levels):
#     Hd = sum_two_body_ZZ(N, J, open_chain)
#     Xsum = sum_one_body_X(N)

# ===== ENERGY LEVELS: E_k(t) vs t =====
def energy_spectrum(
    N=4, J=1.0, h=0.2, T=5.0, alpha=1.0, open_chain=True,
    num_points=200, k_levels=5
):
    Hd   = sum_two_body_ZZ(N, J, open_chain)
    Xsum = sum_one_body_X(N)

    t_grid = np.linspace(0.0, T, num_points)
    levels = np.zeros((k_levels, num_points), dtype=float)

    for i, t in enumerate(t_grid):
        Ht = A_of_t(t, T, alpha)*Hd + B_of_t(t, T, alpha)*(h*Xsum)
        evals, _ = eigh(Ht)              # eigenvalues tăng dần
        k = min(k_levels, len(evals))
        levels[:k, i] = np.real(evals[:k])

    return t_grid, levels

def plot_energy_levels(
    N=4, J=1.0, h=0.2, T=5.0, alpha=1.0, open_chain=True,
    num_points=200, k_levels=8, save_png="energy_levels.png", save_csv="energy_levels.csv"
):
    t_grid, levels = energy_spectrum(N, J, h, T, alpha, open_chain, num_points, k_levels)

    # Lưu CSV: t, E0, E1, ...
    header = ["t"] + [f"E{i}" for i in range(levels.shape[0])]
    data = np.column_stack([t_grid] + [levels[i] for i in range(levels.shape[0])])
    np.savetxt(save_csv, data, delimiter=",", header=",".join(header), comments="")
    print(f"Saved spectrum CSV to {save_csv}")

    plt.figure()
    for i in range(levels.shape[0]):
        plt.plot(t_grid, levels[i], label=rf"$E_{i}(t)$")
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.title(" $H(t)$")

    # Hộp thông số
    ax = plt.gca()
    info = [
        f"N = {N}",
        f"J = {J}",
        f"h = {h}",
        f"T = {T}",
      rf"$s(t)=(t/T)^{{\alpha}}$,$\alpha={alpha}$"
        # rf"s(t)=(t/T)^{{\alpha}},  \alpha = {alpha}",
        # r"A(t)=1-s(t),  B(t)=s(t)"
    ]
    ax.text(0.02, 0.98, "\n".join(info), transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.15))

    plt.legend(loc="best", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    print(f"Saved plot to {save_png}")

    # (tuỳ chọn) in khe năng lượng tối thiểu giữa E1 và E0
    if levels.shape[0] >= 2:
        gap = levels[1] - levels[0]
        j = int(np.argmin(gap))
        print(f"Min gap Δ = E1-E0 ≈ {gap[j]:.6f} at t ≈ {t_grid[j]:.6f}")

#----------- Energy Distribution ----------

def hadamard():
    return (1/np.sqrt(2)) * np.array([[1, 1],[1, -1]], dtype=complex)

    




# ---------- Core: QFI during annealing ----------


def qfi_during_anneal(N=4, J = 1.0, h = 0.2, T =10.0, K = 1000, alpha=1.0, open_chain = True, return_trace=True):
    Hd = sum_two_body_ZZ(N, J, open_chain) 
    Xsum = sum_one_body_X(N)
    E0, psi0 = ground_state(Hd)

    dt = T/K
    dim = 2**N
    U = np.eye(dim, dtype=complex)
    G = np.zeros((dim, dim), dtype=complex)

    times = []
    FQs = []
    FQs_div_t2 = []
    for k in range(K):
        t = k*dt
        tm = t + 0.5*dt
        H = A_of_t(t, T, alpha)*Hd + B_of_t(t, T, alpha)* (h*Xsum)
        # Hmid is not defined in original; use midpoint Hamiltonian
        Hmid = A_of_t(tm, T, alpha)*Hd + B_of_t(tm, T, alpha)* (h*Xsum)
        Ustep = expm(-1j * Hmid * dt)
        dHdh = B_of_t(t, T, alpha)* Xsum
        G += (U.conj().T @ dHdh @ U) * dt 
        U = Ustep @ U

        if return_trace and ( ( k % max (1,K//200)) == 0):
            mean = np.vdot(psi0, G @ psi0)
            var = np.vdot(psi0, (G @ G) @ psi0) - (mean*np.conj(mean))
            FQ = 4.0 * np.real(var)
            times.append(t)
            FQs.append(FQ)

            if t > 0:
                FQs_div_t2.append(FQ / (t**2))
            else:
                FQs_div_t2.append(np.nan)

    # Final QFI at time T
    mean = np.vdot(psi0, G @ psi0)
    var = np.vdot(psi0, (G @ G) @ psi0) - (mean*np.conj(mean))
    FQ_T = 4.0 * np.real(var)
    if return_trace:
        return FQ_T, (np.array(times), np.array(FQs), np.array(FQs_div_t2))
    else:
        return FQ_T, None

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import csv

    # ===== (1) Tham số bạn đang chạy =====
    N = 4
    J = 1.0
    h = 0.2
    T = 100
    # steps_per_unit = pick_steps_per_unit(N, J, h, safety=0.1)
    # K = steps_per_unit * T
    K = 2000
    alpha = 1.0
    schedule_name = "power"  # đúng với s(t) = (t/T)^alpha trong code của bạn

    # ===== (2) Chạy QFI và lấy vệt FQ(t) =====
    FQ_T, trace = qfi_during_anneal(N=N, J=J, h=h, T=T, K=K, alpha=alpha, return_trace=True)
    print(f"Final QFI at T = {T:.3f}: {FQ_T:.8f}")

    # Ghi qfi_trace.csv như cũ
    if trace is not None:
        times, FQs, FQs_div_t2 = trace
        data = np.column_stack([times, FQs, FQs_div_t2])
        np.savetxt("qfi_trace.csv", data, delimiter=",", header="t,FQ,FQ_div_t2", comments="")
        print("Saved qfi_trace.csv with columns: t, FQ, FQ_div_t2")

    # ===== (3) Ghi thêm schedule_trace.csv: A(t), B(t) =====
    # Dùng cùng “sampling” thời gian như qfi_trace để so khớp khi vẽ
    if trace is not None:
        with open("schedule_trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "A", "B"])
            for t in times:
                A_t = A_of_t(t, T, alpha)
                B_t = B_of_t(t, T, alpha)
                w.writerow([t, A_t, B_t])
        print("Save schedule trace to schedule_trace.csv")



    # ===== (4) VẼ FQ(t) + in đầy đủ thông số lên hình =====
    # (Bạn thích nền tối: có thể bật dòng dưới)
    # plt.style.use("dark_background")
    mask = ~np.isnan(FQs_div_t2)
    plt.figure()
    # if trace is not None:
    plt.plot(times[mask], FQs_div_t2[mask], lw=2)
    # else:
        # nếu không lấy trace, vẽ điểm cuối để có hình tối thiểu
        # plt.plot([T], [FQ_T], "o")

    plt.xlabel("t")
    # plt.ylabel(r"$F_Q(t)$")
    plt.ylabel(r"$F_Q(t)/t^2$")

    # CRB: sigma_h >= 1/sqrt(FQ(T))
    # sigma_h = (1.0 / np.sqrt(FQ_T)) if FQ_T > 0 else float("inf")

    # Tiêu đề hiển thị kết quả chính
    # plt.title(rf"$F_Q(T) = {FQ_T:.6f}$,   "
    #           rf"$\sigma_h \geq 1/\sqrt{{F_Q}} \approx {sigma_h:.4f}$")
    plt.title("Quantum Fisher Information vs. Time")

    # Hộp thông tin thông số + lịch
    # s(t) của bạn: (t/T)^alpha  =>  A(t)=1-s(t), B(t)=s(t)
    info_lines = [
        f"N = {N}", 
        f"J = {J}",
        f"h = {h}",
        f"T = {T}",
        f"K = {K}",
        # f"s(t) = (t/T)^{{\alpha}},
        #  \alpha = {alpha}",
        r"A(t) = 1 - s(t)",
        r"B(t) = s(t)",
        rf"$s(t)=(t/T)^{{\alpha}}$,$\alpha={alpha}$"
        # rf" $s(t)=(t/T)^{{\alpha}}$"
    ]
    ax = plt.gca()
    ax.text(0.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round", alpha=0.15))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qfi_vs_t_annotated_div_t2.png", dpi=160)
    print("Saved annotated plot to qfi_vs_t_annotated_div_t2.png")

    # ... sau khi bạn đã có: times, FQs, N,J,h,T,K,alpha và đã import matplotlib.pyplot as plt
    plot_energy_levels(N=4, J=1.0, h=0.2, T=10.0, alpha=1.0,
                       num_points=300, k_levels=8,
                       save_png="energy_levels.png",
                       save_csv="energy_levels.csv")
# === Figure 2: FQ(t) (QFI gốc) ===
plt.figure()

# nếu đã có times, FQs từ trace:
plt.plot(times, FQs, lw=2)

plt.xlabel("t")
plt.ylabel(r"$F_Q(t)$")
plt.title("Quantum Fisher Information vs. Time")
# plt.title(r"$F_Q(t)$ during annealing")

# Reuse info box (1 cột)
ax2 = plt.gca()
info_lines2 = [
    f"N = {N}",
    f"J = {J}",
    f"h = {h}",
    f"T = {T}",
    f"K = {K}",
    r"A(t) = 1 - s(t)",
    r"B(t) = s(t)",
    rf"$s(t)=(t/T)^{{\alpha}}$,$\alpha={alpha}$"
]
ax2.text(0.02, 0.98, "\n".join(info_lines2),
         transform=ax2.transAxes, ha="left", va="top",
         fontsize=10, bbox=dict(boxstyle="round", alpha=0.15))

plt.grid(True)
plt.tight_layout()
plt.savefig("qfi_vs_t_annotated_FQ.png", dpi=160)
print("Saved annotated plot to qfi_vs_t_annotated_FQ.png")


plt.figure()
ax = plt.gca()

# --- FQ(t) trên trục trái ---
ax.plot(times, FQs, lw=2, label=r"$F_Q(t)$")
ax.set_xlabel("t")
ax.set_ylabel(r"$F_Q(t)$")

# --- Trục phụ bên phải cho A(t), B(t) ---
ax2 = ax.twinx()
A_vals = [A_of_t(t, T, alpha) for t in times]
B_vals = [B_of_t(t, T, alpha) for t in times]
ax2.plot(times, A_vals, ls="--", label=r"$A(t)$")
ax2.plot(times, B_vals, ls=":",  label=r"$B(t)$")
ax2.set_ylabel(r"$A(t),\ B(t)$")

# --- Tiêu đề + CRB ---
FQ_T = float(FQs[-1])
sigma_h = (1.0 / np.sqrt(FQ_T)) if FQ_T > 0 else float("inf")
plt.title(rf"$F_Q(T) = {FQ_T:.6f}$,   $\sigma_h \geq 1/\sqrt{{F_Q}} \approx {sigma_h:.4f}$")
plt.title("Quantum Fisher Information vs. Time")

# --- Info box thông số ---
s_expr = rf"s(t) = (t/T)^{{\alpha}},\ \alpha={alpha}"  # vì bạn đang dùng power-law
info_lines = [
    rf"N={N},\ J={J},\ h={h}",
    rf"T={T},\ K={K},\ schedule=power",
    r"A(t)=1-s(t),\ B(t)=s(t)",
    s_expr,
]
ax.text(0.02, 0.98, "\n".join(info_lines),
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, bbox=dict(boxstyle="round", alpha=0.15))

# --- Gộp legend từ cả hai trục ---
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="lower right")

ax.grid(True)
plt.tight_layout()
plt.savefig("qfi_FQ_At_Bt.png", dpi=200)
print("Saved figure: qfi_FQ_At_Bt.png")

