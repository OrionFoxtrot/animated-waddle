import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,newx,newy):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    ax.plot(newx, newy, linewidth=1,color='red')
    plt.show()
    plt.figure()


def GenerateDataPoints(t):  # t is number of points
    x = np.linspace(0, 2 * np.pi, t)
    y = np.sin(x)
    return (x, y)

def GenerateMatrixs(x,t,m,k,y):

    Aa = np.zeros(shape=(t, m))
    Aa[:, 0] = 1
    for rows in range(t):
        for cols in range(1, m):
            Aa[rows, cols] = x[rows] ** cols

    Ab = np.zeros(shape=(t, k))

    #Ab[:, 0] = -y
    for rows in range(t):
        for cols in range(0, k):
            Ab[rows, cols] = (x[rows] ** cols) * -y[rows]

    A = np.hstack((Aa,Ab))
    #New approach Hstack them

    return A

if __name__ == '__main__':
    print("HLST")
    t=20 #Step size = 20
    [x,y] = GenerateDataPoints(t) #Generate the sine wave


    #t = 4; #Test Code
    #x = np.array([-1,1,2,3])
    #y = np.array([1/2,-1,-1/2,3/2]); #End Test
    #print(x)
    #print(y)


    #fig, ax = plt.subplots()
    #ax.plot(x, y, linewidth=2.0)
    #plt.show()

    #m is numerator power
    #k is denominator power
    m = 4
    k = 1
    #A Vek is t x (m+k)

    A = GenerateMatrixs(x,t,m,k,y)
    val,vek = np.linalg.eigh(A.T@A)

    a = vek[0:m,0]
    a = a[::-1]

    b = vek[m:m+k,0]
    b = b[::-1]

    polya = np.poly1d(a.flatten())
    polyb = np.poly1d(b.flatten())
    print(a)
    print(b)
    newy = polya(x)/polyb(x)
    plot(x,y,x,newy)
    #plot(0,0,x,newy)


