import numpy as np
from scipy.linalg import expm, eigh

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

def hadamard():
    return (1/np.sqrt(2)) * np.array([[1, 1],[1, -1]], dtype=complex)

def kronN_list(mats):
    M = mats[0]
    for k in range(1, len(mats)):
        M = np.kron(M, mats[k])
    return M

def psi_at_time(N=4, J=1.0, h=0.2, T=10.0, K=4000, alpha=1.0, open_chain=True, t_eval=None):
    """
    Trả về trạng thái |psi(t_eval)> bằng tiến hoá midpoint với H(t)=A(t)Hd + B(t) h Xsum.
    Nếu t_eval=None -> t_eval=T.
    """
    Hd   = sum_two_body_ZZ(N, J, open_chain)
    Xsum = sum_one_body_X(N)
    _, psi0 = ground_state(Hd)

    if t_eval is None: t_eval = T
    t_eval = float(t_eval)
    dt = T / K
    steps = int(np.floor(t_eval / dt))

    psi = psi0.copy()
    for k in range(steps):
        t  = k * dt
        tm = t + 0.5 * dt
        Hmid  = A_of_t(tm, T, alpha) * Hd + B_of_t(tm, T, alpha) * (h * Xsum)
        Ustep = expm(-1j * Hmid * dt)
        psi   = Ustep @ psi

    # nếu t_eval không khớp bội dt, làm 1 bước còn lại với dt' nhỏ hơn
    rem = t_eval - steps*dt
    if rem > 1e-15:
        tm  = steps*dt + 0.5*rem
        Hmid  = A_of_t(tm, T, alpha) * Hd + B_of_t(tm, T, alpha) * (h * Xsum)
        Ustep = expm(-1j * Hmid * rem)
        psi   = Ustep @ psi

    # chuẩn hoá đề phòng sai số số học
    psi = psi / np.linalg.norm(psi)
    return psi, Hd, Xsum

def energy_distribution_Hc_at_time(N=4, J=1.0, h=0.2, T=10.0, K=4000, alpha=1.0, t_eval=10.0):
    """
    Tính phân bố năng lượng của H_c = sum X_j trên |psi(t_eval)>.
    Trả về:
      vals  : các giá trị riêng có thể có của H_c (−N, −N+2, ..., N)
      probs : xác suất tương ứng
      meanE : <psi| H_c |psi> (kỳ vọng)
    """
    psi_t, Hd, Xsum = psi_at_time(N, J, h, T, K, alpha, t_eval)

    # Biểu diễn trong cơ sở X: áp Hadamard trên mỗi qubit
    H1 = hadamard()
    HN = kronN_list([H1]*N)  # H^{⊗N}
    psi_X = HN @ psi_t       # biên độ trong cơ sở |+/->

    # Xác suất trên từng bitstring b (0: |+>, 1: |->). Giá trị riêng H_c = sum s_i, s_i∈{+1,-1}
    dim = 2**N
    probs_b = np.abs(psi_X)**2
    # map bitstring -> eigenvalue
    eigvals = np.empty(dim, dtype=int)
    for b in range(dim):
        # số bit 1 = số |-> (eigen −1); số bit 0 = số |+> (eigen +1)
        minus = bin(b).count("1")
        plus  = N - minus
        eigvals[b] = plus - minus  # ∑ s_i = (#plus) - (#minus) ∈ {−N,−N+2,...,N}

    # gom histogram theo giá trị riêng
    possible_vals = np.arange(-N, N+1, 2)  # bước 2
    probs = np.zeros_like(possible_vals, dtype=float)
    for b in range(dim):
        idx = int((eigvals[b] + N)//2)  # ánh xạ {−N,...,N} bước 2 -> {0,...,N}
        probs[idx] += probs_b[b].real

    # chuẩn hóa đề phòng trôi số
    probs = probs / probs.sum()

    # kỳ vọng = ∑ E * p(E) (nên trùng với <Xsum>)
    meanE = float(np.dot(possible_vals, probs))

    # kiểm chứng nhanh <psi|Xsum|psi>
    mean_direct = float(np.vdot(psi_t, (sum_one_body_X(N) @ psi_t)).real)

    return possible_vals, probs, meanE, mean_direct

# === Ví dụ chạy và lưu CSV + vẽ bar chart ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N, J, h = 4, 1.0, 0.2
    T, K, alpha = 10.0, 4000, 1.0
    t_eval = 10.0  # tính tại t = 10

    vals, probs, meanE, mean_direct = energy_distribution_Hc_at_time(
        N=N, J=J, h=h, T=T, K=K, alpha=alpha, t_eval=t_eval
    )

    print(f"<H_c> from histogram  = {meanE:.6f}")
    print(f"<H_c> direct (⟨Xsum⟩) = {mean_direct:.6f}")

    # Lưu CSV: E, p(E)
    out = np.column_stack([vals, probs])
    np.savetxt("energy_dist_t10.csv", out, delimiter=",", header="E,p", comments="")
    print("Saved energy_dist_t10.csv")

    # Vẽ histogram rời rạc theo các mức −N, −N+2, ..., N
    plt.figure()
    plt.bar(vals, probs, width=0.8, align="center")
    plt.xlabel(r"Eigenvalue $E$ of $H_c=\sum_j X_j$")
    plt.ylabel("Probability")
    plt.title(rf"Energy distribution of $H_c$ at $t={t_eval}$  (N={N}, T={T}, \alpha={alpha})")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("energy_dist_t10.png", dpi=200)
    plt.show()
