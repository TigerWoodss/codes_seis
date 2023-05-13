import networkx as nx
import numpy as np
from numba import njit
import pandas as pd
import matplotlib.pyplot as plt
import time


@njit
def generate_network(n, m, br, alpha):
    """
    Generate adjacency matrix and connect active nodes to fixed nodes with edges

    Args:
        n: umber of fixed nodes per layer of the static network
        m: number of active nodes
        br: adjacency matrix of the network used in each layer of the static network
        alpha: the probability that an active node is connected to an edge of a fixed node


    Returns:
        strc: np.array  (n+m)*(n+m)
              Adjacency matrix of the entire multilayer network
    """
    strc = np.zeros((n+m, n+m), dtype=np.int8)
    strc[:n, :n] = br

    for x in range(n):
        for y in range(m):
            p = np.random.random()
            if p < alpha:
                strc[x, n+y] = 1
                strc[n+y, x] = 1

    return strc


def layer_active(strc, actives, layernum):
    layer = np.copy(strc)
    act = np.argwhere(actives == layernum)
    layer[act] = 0
    layer[:, act] = 0
    return layer


@njit
def infect(layer, status, actives, d, n, beta, t_etoi, t_itos):
    """
   Nodes Infection

    Args:
        layer: adjacency matrix of the activated static network layer
        status: the state matrix of the node
        actives: matrix of the number of layers to which the active node belongs
        D: the number of layers of the static network
        n: the number of fixed nodes in each layer of the static network
        beta: probability of node being infected
        t_etoi: the time step required to change from state e to state i
        t_itos: the time step required to change from state i to state s

    Returns:
        new_status: state matrix of the node after infection
    """
    new_status = np.copy(status)
    nons = np.argwhere(status[d] > 0)
    act = np.argwhere(actives == d)
    for i in act.flat:
        if status[-1, i] > 0:
            adj = np.flatnonzero(layer[i+n])
            for j in adj:
                if new_status[d, j] == 0 and status[d, j] == 0:
                    p = np.random.random()
                    if p < beta:
                        new_status[d, j] = 1
            if status[-1, i] == 2:
                p = np.random.random()
                if p < 1/t_itos:
                    new_status[-1, i] = 0
            else:
                p = np.random.random()
                if p < 1/t_etoi:
                    new_status[-1, i] = 2
    for i in nons.flat:
        if status[d, i] == 2:
            p = np.random.random()
            if p < 1/t_itos:
                new_status[d, i] = 0
        else:
            p = np.random.random()
            if p < 1/t_etoi:
                new_status[d, i] = 2
        adj = np.flatnonzero(layer[i])
        for j in adj:
            if j < n:
                x = d
                y = j
            else:
                x = -1
                y = j - n
            if(new_status[x, y] == 0 and status[x, y] == 0):
                p = np.random.random()
                if p < beta:
                    new_status[x, y] = 1

    return new_status


clk = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
print("start   {}".format(clk))

D = 5
n = 10000
m = 10
source = 10
# ms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
#       200, 300, 400, 500, 600, 700, 800, 1000]
iteration = 500
# beta = 0.02
betas = [i/100 for i in range(1, 101)]
t_etoi = 7
t_itos = 21
alpha = 0.04

br = nx.to_numpy_array(nx.barabasi_albert_graph(n, 1))
actives = np.zeros(m, dtype=np.int8)
strc = generate_network(n, m, br, alpha)

iters = np.arange(iteration)
for beta in betas:
    print("===================== beta:{} =====================".format(beta))
    status = np.zeros((D+1, n), dtype=np.int8)
    seed = np.random.choice(n, 10, replace=False)
    # print(seed)
    status[0, seed] = 1
    print(np.flatnonzero(status[0]))

    inums = []
    enums = []
    for iter in range(iteration):
        actives = np.random.randint(D, size=m)

        for d in range(D):
            layer = layer_active(strc, actives, d)
            status = infect(layer, status, actives, d, n, beta, t_etoi, t_itos)

        enums.append(status[status == 1].size)
        inums.append(status[status == 2].size)
        iscales = np.array(inums)/(5*n+m)
        escales = np.array(enums)/(5*n+m)

    clk = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("infect finish   {}".format(clk))
    data = {
        "inums": inums,
        "enums": enums
    }
    df = pd.DataFrame(data)
    df.to_csv("./result/beta_10/data/beta{}.csv".format(beta))
    plt.figure(figsize=(150, 75))
    plt.plot(iters, iscales, 'r', marker='.', label='inums')
    plt.plot(iters, escales, '--', 'b', marker='.', label='enums')
    plt.legend(['inums', 'enums'], loc='upper left', fontsize=80)
    plt.title("actives={},beta={},alpha={},t_etoi={},t_itos={}".format(
        m/n, beta,  alpha, t_etoi, t_itos), fontsize=80)
    plt.axis([0, 500, 0, 1])
    plt.savefig("./result/beta_10/img/beta{}.jpg".format(beta))
    plt.close()

    clk = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("finish   {}".format(clk))
