import numpy as np
from scipy.optimize import minimize
from qutip import \
        Qobj, basis, qeye, \
        sigmax, sigmay, sigmaz, \
        tensor, expect, concurrence

def density_matrix_mle(counts_4x4):
    """
    Perform quantum state tomography for a 2-qubit system
    using Maximum Likelihood Estimation (MLE), given measurement counts.

    Parameters
    ----------
    counts_4x4 : np.ndarray, shape (4,4)
        2D array of raw counts from measurement outcomes.
        Order: [(H,H), (H,V), (V,V), (V,H), ...] and 12 additional outcomes,
        row-major.

    Returns
    -------
    rho_mle_q : Qobj (4×4)
        The estimated MLE density matrix.
    purity : float
        Tr(rho^2).
    conc : float
        Wootters' concurrence.
    """
    # --- Define Pauli Matrices and Gamma Construction ---
    paulis = [qeye(2), sigmax(), sigmay(), sigmaz()]
    Gamma = [0.5 * tensor(p1, p2) for i, p1 in enumerate(paulis)
             for j, p2 in enumerate(paulis) if (i, j) != (0, 0)]
    Gamma.append(0.5 * tensor(paulis[0], paulis[0]))

    # --- Define Tomographic Projectors ---
    H = basis(2, 0)
    V = basis(2, 1)
    D = (H + V).unit()
    R = (H + 1j * V).unit()
    L = (H - 1j * V).unit()

    # All 16 measurement projectors (following common order)
    states = [tensor(a, b) for a, b in [
        (H, H), (H, V), (V, V), (V, H),
        (R, H), (R, V), (D, V), (D, H),
        (D, R), (D, D), (R, D), (H, D),
        (V, D), (V, L), (H, L), (R, L)
    ]]
    projectors = [s * s.dag() for s in states]

    # --- Reshape counts and process probabilities ---
    counts = np.asarray(counts_4x4, dtype=float).flatten() # shape (16,)
    N = np.sum(counts[:4])  # total number of counts in the first basis (optional)
    ps = counts / N         # normalized probabilities

    # --- Build B and M Matrices (see James et al. 2001) ---
    B = np.array([[G.overlap(P) for G in Gamma] for P in projectors])
    B_inv = np.linalg.inv(B)
    Ms = [sum(B_inv[m, n] * Gamma[m] for m in range(16)) for n in range(16)]

    # --- Linear inversion (starting guess for MLE) ---
    rho_lin = sum(M * p for M, p in zip(Ms, ps))

    # --- Cholesky Parameterization ---
    # (Alternative way to work with T)
    def T_from_params(t):
        """Reconstruct lower-triangular Cholesky matrix from parameters."""
        T = np.zeros((4, 4), dtype=complex)
        diag = [0, 1, 2, 3]
        lower = [(1, 0), (2, 1), (2, 0), (3, 1), (3, 0), (3, 2)]
        T[0, 0], T[1, 1], T[2, 2], T[3, 3] = t[0:4]
        for i, (r, c) in enumerate(lower):
            T[r, c] = t[4 + 2*i] + 1j * t[5 + 2*i]
        # Set dims to [[2,2],[2,2]] for 2-qubit
        return Qobj(T, dims=[[2,2], [2,2]])

    def rho_from_t(t):
        """Build positive semidefinite, normalized density matrix from params."""
        T = T_from_params(t)
        rho = T.dag() * T
        return rho / rho.tr()

    # MLE expression to optimize
    def neg_log_likelihood(t):
        """
        Negative log-likelihood (actually a Poisson/chi2 approximation)
        for measurement counts, for optimizer.
        """
        rho = rho_from_t(t)
        probs = np.real([expect(P, rho) for P in projectors])
        # To avoid division by zero
        expected = np.clip(probs * N, 1e-8, None)
        return np.sum((expected - counts)**2 / (2 * expected))

    # --- Initial Cholesky guess from linear inversion ---
    eigvals = rho_lin.eigenenergies()
    min_eig = np.min(eigvals)
    # 4x4 identity with dims [[2, 2], [2, 2]
    identity = tensor(qeye(2), qeye(2))
    # Adjust to positive semidefinite if needed
    rho0 = rho_lin + max(-min_eig + 1e-6, 0) * identity
    rho0 = rho0 / rho0.tr()

    # Obtaining lower L on Cholesky decomposition $\mathbf {A} =\mathbf {LL} ^{*}$
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
    T0 = np.linalg.cholesky(rho0.full())

    t0 = np.zeros(16)
    # Collect the diagonal terms in t0
    t0[0:4] = np.real(np.diag(T0))
    # Lower triangle elements
    lower_indices = [(1, 0), (2, 1), (2, 0), (3, 1), (3, 0), (3, 2)]
    # Collect the triangle elements in t0 as
    #..., Re(T0[r, c]), Im(T0[r, c]), ...
    for i, (r, c) in enumerate(lower_indices):
        t0[4 + 2*i] = np.real(T0[r, c])
        t0[5 + 2*i] = np.imag(T0[r, c])

    # --- Optimization (MLE) ---
    res = minimize(neg_log_likelihood, t0, method='BFGS')
    # rho_mle is a Qobj [[2, 2], [2, 2]]
    rho_mle = rho_from_t(res.x)
    # rho_mle.full() --> Return as plain numpy array (shape 4x4)

     # compute metrics
    purity = float(rho_mle.purity())
    conc   = float(concurrence(rho_mle))

    return rho_mle.full(), purity, conc

# Example usage:
if __name__ == "__main__":
    # James et al, 2001 data
    counts = np.array([34749, 324, 35805, 444, 16324, 17521, 13441, 16901,
    17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586])
    rho, pu, co = density_matrix_mle(counts)
    # Number format
    np.set_printoptions(
    precision=4,
    suppress=True,
    formatter={
        'complexfloat': lambda x:
            f"{x.real:.4f}"
            if abs(x.imag) < 1e-10
            else f"{x.real:.4f}{x.imag:+.4f}j"
    }
)
    print("ρ_MLE:\n", rho)
    print(f"Purity:      {pu:.4f}")
    print(f"Concurrence: {co:.4f}")

