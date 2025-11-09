import numpy as np
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt
import pandas as pd

# ===== (1) CÁC HÀM TIỆN ÍCH (Giống như code gốc) =====

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
    # Hermitize để đảm bảo ổn định số
    H_herm = (H + H.conj().T) / 2
    E, V = eigh(H_herm)
    return E[0], V[:,0]

def s_of_t(t, T, alpha=1.0):
    s = (t/T) ** alpha
    return min(max(s, 0.0),1.0)

def A_of_t(t, T, alpha=1.0):
    return 1.0 - s_of_t(t, T, alpha)

def B_of_t(t, T, alpha=1.0):
    return s_of_t(t, T, alpha)

# ===== (2) HÀM TIẾN HÓA TRẠNG THÁI =====
def evolve_final_state(N, J, h, T, K, alpha):
    """
    Tiến hóa trạng thái từ t=0 đến t=T và trả về trạng thái cuối cùng |psi(T)>.
    """
    # Xây dựng Hamiltonian
    Hd = sum_two_body_ZZ(N, J, open_chain=True) 
    Xsum = sum_one_body_X(N)
    Hp = h * Xsum # H_p trong H(t) có chứa 'h'
    
    # Trạng thái ban đầu |psi(0)>
    E0, psi_t = ground_state(Hd)
    psi_t = psi_t.astype(complex) # Đảm bảo là complex

    dt = T/K
    
    print(f"Bắt đầu tiến hóa trạng thái: N={N}, T={T}, K={K}...")
    for k in range(K):
        tm = (k + 0.5) * dt # Thời gian điểm giữa
        
        # Hamiltonian ở điểm giữa
        Hmid = A_of_t(tm, T, alpha) * Hd + B_of_t(tm, T, alpha) * Hp
        
        # Bước tiến hóa U_step = exp(-i * H_mid * dt)
        Ustep = expm(-1j * Hmid * dt)
        
        # Cập nhật trạng thái |psi(t+dt)> = U_step |psi(t)>
        psi_t = Ustep @ psi_t
        
    print("Tiến hóa hoàn tất.")
    return psi_t # Trạng thái cuối cùng |psi(T)>

# ===== (3) SCRIPT CHÍNH ĐỂ TÍNH PHÂN BỐ =====
if __name__ == "__main__":
    
    # --- (A) Tham số mô phỏng (T=10) ---
    N = 4
    J = 1.0
    h = 0.2
    T = 100    # Thời gian cuối theo yêu cầu
    K = 50000    # Số bước thời gian
    alpha = 1.0 

    # --- (B) Bước 1: Lấy trạng thái cuối cùng |psi(T)> ---
    psi_T = evolve_final_state(N, J, h, T, K, alpha)
    
    # --- (C) Bước 2: Xây dựng H_c và tìm phổ của nó ---
    # H_c là "tổng chuỗi pauli X", không có 'h'
    # Đây là Hamiltonian mà chúng ta muốn đo
    Hc = sum_one_body_X(N)
    
    # Giá trị kỳ vọng <E> = <psi|Hc|psi> (một con số duy nhất)
    E_expect = np.vdot(psi_T, Hc @ psi_T)
    print(f"\nGiá trị kỳ vọng <psi(T)| Hc |psi(T)> = {E_expect.real:.6f}")

    # Tìm các mức năng lượng (giá trị riêng) E_k và
    # các trạng thái năng lượng (vectơ riêng) |e_k> của Hc
    # energies_c: mảng các E_k
    # states_c: ma trận với các cột là các |e_k>
    print("Tìm phổ của Hc = sum(X)...")
    energies_c, states_c = eigh(Hc)

    # --- (D) Bước 3: Tính xác suất P(E_k) = |<e_k | psi(T)>|^2 ---
    
    # states_c.conj().T là ma trận với các hàng là <e_k| (bra)
    # Phép nhân ma trận (@) này thực hiện tất cả các phép chiếu cùng lúc
    # prob_vector[k] = |<e_k | psi_T>|^2
    prob_vector = np.abs(states_c.conj().T @ psi_T)**2
    
    # --- (E) Xử lý và nhóm các kết quả ---
    # Do Hc có thể có suy biến (nhiều trạng thái |e_k> có cùng năng lượng E_k)
    # Chúng ta cần cộng tất cả các xác suất của các trạng thái suy biến đó lại.
    
    prob_distribution = {}
    # Làm tròn năng lượng để nhóm lại (vd: 0.0 và -0.0000001 là một)
    rounded_energies = np.round(energies_c, 5)
    
    unique_energies = np.unique(rounded_energies)
    
    for E_val in unique_energies:
        # Tìm tất cả các index k có cùng mức năng lượng E_val
        indices = np.where(rounded_energies == E_val)
        # Cộng tất cả xác suất P(E_k) tương ứng
        total_prob = np.sum(prob_vector[indices])
        prob_distribution[E_val] = total_prob
        
    print("\nPhân bố xác suất P(E_k) tại T=10:")
    for E, P in prob_distribution.items():
        print(f"  E = {E:5.1f} : P = {P*100:6.2f}%")
        
    # Kiểm tra tổng xác suất (phải bằng 1)
    print(f"  Tổng xác suất: {np.sum(list(prob_distribution.values())):.5f}")

    # --- (F) Vẽ biểu đồ ---
    E_vals = list(prob_distribution.keys())
    Probs = list(prob_distribution.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(E_vals, Probs, width=0.5, align='center', edgecolor='black')
    
    plt.xlabel("Energy $E_k$ (Eigenvalue of $H_c = \sum X_j$)")
    plt.ylabel(r"Probability $P(E_k) = |\langle e_k | \psi(T) \rangle|^2$")
    plt.title(f"Energy Distribution $\psi(T={T})$ in the basis of $H_c$")
    plt.xticks(unique_energies) # Đảm bảo các mức năng lượng được hiển thị
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hộp thông tin
    info_lines = [
        f"N = {N}", f"J = {J}", f"h = {h}",
        f"T = {T}", f"K = {K}", rf"$\alpha$ = {alpha}"
    ]
    ax = plt.gca()
    ax.text(0.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.6))
            
    plt.tight_layout()
    plt.savefig("energy_distribution_T10.png", dpi=150)
    
    print("\nĐã lưu biểu đồ phân bố năng lượng vào 'energy_distribution_T10.png'")
    plt.show() # Hiển thị đồ thị