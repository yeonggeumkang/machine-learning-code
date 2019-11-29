import numpy as np
import matplotlib.pyplot as plt

# rnn_gradient_graph (기울기 크기변화 관찰/p.243)
N = 2 #미니배치 크기
H = 3 #은닉상태 벡터의 차원 수
T = 20 #시계열 데이터의 길이

dh = np.ones((N,H))
np.random.seed(3) #난수 시드 고정
#Wh = np.random.randn(H,H)
Wh = np.random.randn(H,H) * 0.5 #Wh 초깃값 변경

norm_list=[]
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(dh)
print(Wh)
print(norm_list)

#기울기 크리핑 함수 구현(p.246)
dW1 = np.random.rand(3,3) * 10
dW2 = np.random.rand(3,3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for gard in grads:
        total_norm += np.sum(grad **2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm/(total_norm+1e-6)
    if rate<1:
        for grad in grads:
            grad *= rate

# LSTM클래스 구현
class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f=sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g*i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

# TimeSTM class 구현
class TimeSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_likbe(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N,T,H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeors((N,H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

            return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, D = dhs.shape
        D = Wh.shape[0]

        dxs = np.empty((N,T,D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]

        for t in resversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :], dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None
