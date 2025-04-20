def add(a, b):
    return a + b

def fibonacci(n):
    """
    Return the n‑th Fibonacci number (0‑indexed):
      fibonacci(0) == 0
      fibonacci(1) == 1
    """
    if n < 0:
        raise ValueError("n must be a non‑negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# ------------------ advanced math routines ------------------
import cmath
import math

def gamma_lanczos(z):
    """
    Lanczos approximation for the Gamma function.
    """
    # Coefficients for g=7, n=9
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    if z.real < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi*z) * gamma_lanczos(1 - z))
    z -= 1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + len(p) - 0.5
    return math.sqrt(2*math.pi) * t**(z + 0.5) * math.exp(-t) * x

def binomial_coefficient(n, k):
    """
    Compute C(n, k) via Gamma to handle large values.
    """
    return gamma_lanczos(n + 1) / (gamma_lanczos(k + 1) * gamma_lanczos(n - k + 1))

def solve_quadratic(a, b, c):
    """
    Returns the two roots of ax^2 + bx + c = 0 (complex if necessary).
    """
    disc = cmath.sqrt(b*b - 4*a*c)
    return ((-b + disc) / (2*a), (-b - disc) / (2*a))

def prime_factorization(n):
    """
    Returns the list of prime factors of n.
    """
    factors = []
    # handle 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # odd factors
    p = 3
    while p*p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors

def runge_kutta_4(f, y0, t0, t1, steps):
    """
    Classic RK4 integrator for dy/dt = f(y, t).
    Returns list of (t, y).
    """
    h = (t1 - t0) / steps
    t, y = t0, y0
    trajectory = [(t, y)]
    for _ in range(steps):
        k1 = f(y, t)
        k2 = f(y + 0.5*h*k1, t + 0.5*h)
        k3 = f(y + 0.5*h*k2, t + 0.5*h)
        k4 = f(y + h*k3, t + h)
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        trajectory.append((t, y))
    return trajectory

def simpson_integration(func, a, b, n):
    """
    Composite Simpson's rule (n must be even).
    """
    if n % 2:
        raise ValueError("n must be even")
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        coef = 4 if i % 2 else 2
        s += coef * func(a + i*h)
    return s * h/3

def discrete_fourier_transform(x):
    """
    Naive DFT of sequence x (returns list of complex).
    """
    N = len(x)
    return [sum(x[n] * cmath.exp(-2j*math.pi*k*n/N) for n in range(N)) for k in range(N)]

def matrix_determinant(m):
    """
    Recursive determinant for square matrix m (list of lists).
    """
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0]*m[1][1] - m[0][1]*m[1][0]
    det = 0
    for c in range(n):
        # build submatrix
        sub = [row[:c] + row[c+1:] for row in m[1:]]
        det += ((-1)**c) * m[0][c] * matrix_determinant(sub)
    return det

def eigenvalues_2x2(m):
    """
    Analytic eigenvalues of a 2x2 matrix m = [[a,b],[c,d]].
    """
    a, b = m[0]
    c, d = m[1]
    tr = a + d
    det = a*d - b*c
    disc = cmath.sqrt(tr*tr - 4*det)
    return ((tr + disc)/2, (tr - disc)/2)

# ---------------- super–complicated math routines ----------------

def lambert_W(x, tol=1e-12, maxiter=50):
    """Principal branch of Lambert W via Halley's method."""
    if x < -1/math.e:
        raise ValueError("x must be ≥ -1/e")
    # initial guess
    w = math.log1p(x)
    for _ in range(maxiter):
        e = math.exp(w)
        f = w*e - x
        # Halley update
        dw = f / (e*(w+1) - (w+2)*f/(2*(w+1)))
        w -= dw
        if abs(dw) < tol*(1+abs(w)):
            return w
    raise RuntimeError("Lambert W did not converge")

def riemann_zeta(s, terms=100):
    """Riemann zeta via Euler–Maclaurin summation (s≠1)."""
    if s == 1:
        raise ValueError("Pole at s=1")
    # partial sum
    S = sum(1.0/n**s for n in range(1, terms+1))
    # Euler–Maclaurin correction up to B6
    t = terms
    correction = t**(-s+1)/(s-1) + t**(-s)/2
    correction += s*t**(-s-1)/12 - s*(s+1)*(s+2)*t**(-s-3)/720
    return S + correction

def recursive_fft(x):
    """Cooley–Tukey FFT (length must be power of two)."""
    N = len(x)
    if N <= 1:
        return x
    if N & (N-1):
        # not power of two, fall back
        return discrete_fourier_transform(x)
    even = recursive_fft(x[0::2])
    odd  = recursive_fft(x[1::2])
    factor = [cmath.exp(-2j*math.pi*k/N) * odd[k] for k in range(N//2)]
    return [even[k] + factor[k] for k in range(N//2)] + \
           [even[k] - factor[k] for k in range(N//2)]

def qr_decomposition(A):
    """Classical Gram–Schmidt QR decomposition of A (n×m)."""
    n, m = len(A), len(A[0])
    Q = [[0.0]*m for _ in range(n)]
    R = [[0.0]*m for _ in range(m)]
    for j in range(m):
        # v = A[:,j]
        v = [A[i][j] for i in range(n)]
        for i in range(j):
            R[i][j] = sum(Q[k][i] * A[k][j] for k in range(n))
            for k in range(n):
                v[k] -= R[i][j] * Q[k][i]
        norm = math.sqrt(sum(vi*vi for vi in v))
        R[j][j] = norm
        if norm == 0:
            raise ValueError("Matrix has linearly dependent columns")
        for k in range(n):
            Q[k][j] = v[k] / norm
    return Q, R

def elliptic_integrals(m, tol=1e-12):
    """
    Legendre complete elliptic integrals K(m) and E(m)
    via the arithmetic–geometric mean (AGM) method.
    """
    a, b = 1.0, math.sqrt(1 - m)
    c = math.sqrt(m)
    sumE = 1.0
    two_pow = 1.0
    while abs(c) > tol:
        a_next = (a + b) / 2
        b = math.sqrt(a * b)
        c = (a - b) / 2
        two_pow *= 2
        sumE += two_pow * c*c
        a = a_next
    K = math.pi / (2 * a)
    E = K * (1 - sumE/2)
    return K, E

# -----------------------------------------------------------------
