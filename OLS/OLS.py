import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, newx, newy):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0,label="Sine(x)")
    ax.plot(newx, newy, linewidth=1, color='red',label="OLS of Sine(x)")
    plt.title("OLS Regression Sine Wave")
    ax.legend()
    plt.show()


def GenerateDataPoints(t):  # t is number of points
    x = np.linspace(0, 2 * np.pi, t)
    y = np.sin(x)
    return (x, y)


def GenerateMatrixs(x, t, n, y):
    A = np.zeros(shape=(t, n + 1))
    A[:, 0] = 1
    for rows in range(t):
        for cols in range(1, n + 1):
            A[rows, cols] = x[rows] ** cols
    print("A=")
    print(A)
    return A


if __name__ == '__main__':
    print("OLST")
    t = 20  # Step size = 20
    [x, y] = GenerateDataPoints(t)  # Generate the sine wave

    # t = 4; #Test Code
    # x = np.array([-1,1,2,3])
    # y = np.array([[1/2,-1,-1/2,3/2]]); #End Test

    n = 10  # Poly

    # Poly OLS single poly
    # Form being c1 + c1 u + c2 u^2 + cn u^n
    # A is a txn
    # Y is a tx1
    A = GenerateMatrixs(x, t, n, y)

    #sol = np.linalg.inv((A.T @ A)) @ A.T@y.T

    sol, _, _, _ = np.linalg.lstsq(A, y.T, None)
    sol = sol[::-1]

    poly = np.poly1d(sol.flatten())
    newy = poly(x)

    plot(x, newy+0.05, x, y.flatten())

