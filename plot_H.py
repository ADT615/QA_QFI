import numpy as np
from scipy.linalg import eigh # Dùng eigh để tìm giá trị riêng
import matplotlib.pyplot as plt

# ===== (1) CÁC HÀM TIỆN ÍCH (Giống như code gốc) =====
# (Bao gồm: pauli_mats, op_on_site, sum_two_body_ZZ,
#  sum_one_body_X, s_of_t, A_of_t, B_of_t)

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

def s_of_t(t, T, alpha=1.0):
    s = (t/T) ** alpha
    return min(max(s, 0.0),1.0)

def A_of_t(t, T, alpha=1.0):
    return 1.0 - s_of_t(t, T, alpha)

def B_of_t(t, T, alpha=1.0):
    return s_of_t(t, T, alpha)

# ===== (2) SCRIPT CHÍNH ĐỂ VẼ PHỔ NĂNG LƯỢNG =====

if __name__ == "__main__":
    
    # --- (A) Tham số mô phỏng ---
    N = 4       # Số spin (Kích thước hệ 2^N = 16)
    J = 1.0     # Tương tác Ising
    h = 0.2     # Biên độ trường ngang
    T = 10.0    # Tổng thời gian ủ
    alpha = 1.0 # Lịch trình tuyến tính
    
    # --- (B) Tham số cho việc vẽ ---
    num_t_steps = 100  # Số điểm thời gian để tính toán
    num_levels_to_plot = 8 # Số mức năng lượng thấp nhất muốn vẽ (tối đa là 2**N)

    # --- (C) Xây dựng các Hamiltonian cơ sở ---
    print(f"Bắt đầu: N = {N}, J = {J}, h = {h}, T = {T}, alpha = {alpha}")
    print(f"Xây dựng Hamiltonian... Kích thước ma trận: {2**N} x {2**N}")
    Hd = sum_two_body_ZZ(N, J, open_chain=True)
    Xsum = sum_one_body_X(N)
    Hp = h * Xsum # Hamiltonian điều khiển đầy đủ

    # --- (D) Vòng lặp tính toán giá trị riêng ---
    times = np.linspace(0, T, num_t_steps)
    
    # Mảng để lưu trữ tất cả các giá trị riêng
    # Kích thước: (số_bước_thời_gian, 2**N)
    all_eigenvalues = np.zeros((num_t_steps, 2**N)) 

    print("Tính toán phổ năng lượng theo thời gian...")
    for i, t in enumerate(times):
        # Xây dựng H(t) = A(t)*Hd + B(t)*Hp
        A_t = A_of_t(t, T, alpha)
        B_t = B_of_t(t, T, alpha)
        H_t = A_t * Hd + B_t * Hp
        
        # Hermitize (cho ổn định số) và tìm các giá trị riêng
        # eigh trả về mảng các giá trị riêng đã được sắp xếp từ thấp đến cao
        H_t_herm = (H_t + H_t.conj().T) / 2
        eigenvalues = eigh(H_t_herm, eigvals_only=True)
        all_eigenvalues[i, :] = eigenvalues
    
    # --- (E) Vẽ đồ thị ---
    print("Vẽ đồ thị...")
    plt.figure(figsize=(10, 7))
    
    # Chỉ vẽ 'num_levels_to_plot' mức năng lượng thấp nhất
    for i in range(num_levels_to_plot):
        label = f"$E_{i}(t)$"
        if i == 0:
            label += " (Ground State)"
        if i == 1:
            label += " (1st Excited)"
            
        # all_eigenvalues[:, i] là lịch sử của mức năng lượng thứ i
        plt.plot(times, all_eigenvalues[:, i], lw=2, label=label)
        
    plt.xlabel("t (Time)")
    plt.ylabel("Energy E(t)")
    plt.title(f"Phổ Năng lượng H(t) (N={N}, T={T})")
    
    # Đặt chú giải (legend) bên ngoài biểu đồ cho rõ ràng
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    # Hộp thông tin (giống code trước)
    info_lines = [
        f"N = {N}", f"J = {J}", f"h = {h}",
        f"T = {T}", rf"$\alpha$ = {alpha}"
    ]
    ax = plt.gca()
    ax.text(0.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.5))
    
    # Điều chỉnh layout để chừa không gian cho chú giải
    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    plt.savefig("energy_spectrum.png", dpi=150)
    
    print("Đã lưu biểu đồ phổ năng lượng vào 'energy_spectrum.png'")
    plt.show() # Hiển thị đồ thị