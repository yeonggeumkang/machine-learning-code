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

        #slice 이하 생략
