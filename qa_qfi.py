import numpy as np
from numpy import kron
from scipy.linalg import expm, eigh

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

def s_of_t(t, T, alpha=1.0):
    s = (t/T) ** alpha
    return min(max(s, 0.0),1.0)

def A_of_t(t, T, alpha=1.0):
    return 1.0 - s_of_t(t, T, alpha)

def B_of_t(t, T, alpha=1.0):
    return s_of_t(t, T, alpha)

def qfi_during_anneal(N=4, J = 1.0, h = 0.2, T =5.0, K = 1000, alpha=2.0, open_chain = True, return_trace=True):
    Hd = sum_two_body_ZZ(N, J, open_chain) 
    Xsum = sum_one_body_X(N)
    E0, psi0 = ground_state(Hd)

    dt = T/K
    dim = 2**N
    U = np.eye(dim, dtype=complex)
    G = np.zeros((dim, dim), dtype=complex)

    times = []
    FQs = []
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

    # final QFI at time T
    mean = np.vdot(psi0, G @ psi0)
    var = np.vdot(psi0, (G @ G) @ psi0) - (mean*np.conj(mean))
    FQ_T = 4.0 * np.real(var)
    if return_trace:
        return FQ_T, (np.array(times), np.array(FQs))
    else:
        return FQ_T, None

if __name__ == "__main__":
    N = 4
    J = 1.0
    h = 0.2
    T = 10.0
    K = 2000
    alpha = 1.0

    FQ_T, trace = qfi_during_anneal(N=N, J=J, h=h, T=T, K=K, alpha=alpha, return_trace=True)
    print(f"Final QFI at T = {T:.3f}: {FQ_T:.8f}")

    if trace is not None:
        times, FQs = trace
        data = np.column_stack([times, FQs])
        np.savetxt("qfi_trace.csv", data, delimiter=",", header="t,FQ", comments="")
        print("Save QFI time trace to qfi_trace.csv")


