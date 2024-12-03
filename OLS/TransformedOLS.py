import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, newx, newy):
    fig, ax = plt.subplots()
    Real = ax.plot(x, y, linewidth=2.0)
    Est = ax.plot(newx, newy, linewidth=1, color='red')
    ax.legend(["Real", "Est"])
    plt.show()


def GenerateDataPoints(t):  # t is number of points
    x = np.linspace(0, 2 * np.pi, t)
    y = np.sin(x)
    return (x, y)

def GenerateMatrixs(x,t,m,k,y):

    A = np.zeros(shape=(t,m+1+k))
    Aa = A[:,:m+1]
    Ab = A[:,m+1:]
    for rows in range(t):
        for cols in range(0, m+1):
            Aa[rows, cols] = x[rows] ** cols
    for rows in range(t):
        for cols in range(0, k):
            Ab[rows, cols] = (x[rows] ** (cols+1)) * -y[rows]

    return A


def main():
    t = 20  # Step size = 20
    [x, y] = GenerateDataPoints(t)  # Generate the sine wave

    #t = 4; #Test Code
    #x = np.array([-1,1,2,3])
    #y = np.array([1/2,-1,-1/2,3/2]); #End Test
    #print(x)
    #print(y)

    # fig, ax = plt.subplots()
    # ax.plot(x, y, linewidth=2.0)
    # plt.show()

    # m is numerator power
    # k is denominator power
    m = 3
    k = 1


    A = GenerateMatrixs(x, t, m, k, y)
    #print(A,y)

    sol, _, _, _ = np.linalg.lstsq(A, y, None)

    a = sol[0:m+1]
    a = a[::-1]
    print("coef a:", a)
    b = sol[m+1:]
    b = b[::-1]
    b=np.append(arr=b,values=1)
    print("coef b:",b)


    polya = np.poly1d(a.flatten())
    polyb = np.poly1d(b.flatten())

    newy = polya(x) / polyb(x)
    plot(x, y, x, newy)



if __name__ == '__main__':
    main()