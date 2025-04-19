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

# -----------------------------------------------------------------
