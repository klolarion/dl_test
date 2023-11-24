import tensorflow as tf
import numpy as np


# 자동미분
#  x1 = 3, x2 = 1 에서  y = (x1 + 2 * x2) ** 2 의 편미분  dx_x1, dx_x2 를 구하는 코드
x1 = tf.Variable(3.)  # trainable 기본값 True
# x2 = tf.Variable(1., trainable=False)
x2 = tf.Variable(1.)

with tf.GradientTape() as t:  # GradientTape을 만들고 x2 추적 시작. trainable이 True인 경우 추적 불필요
#     t.watch(x2)
    y = (x1 + 2 * x2) ** 2  # 정방향 진행 연산(미분 대상)
dy_dx = t.gradient(y, [x1, x2])
print(f'dy/dx1 = {dy_dx[0]}')
print(f'dy/dx2 = {dy_dx[1]}')


# 자동미분을 이요한 선형 회귀 학습
x = tf.constant([1., 3., 5., 7.])
y = tf.constant([2., 3., 4., 5.])
w = tf.Variable(1.)  # 가중치
b = tf.Variable(0.5)  # 바이어스
learning_rate = 0.01  # 학습률
epochs = 1000  # 반복횟수


#  입력 x와 레이블 y에 대한 학습
def train_step(x, y):
    with tf.GradientTape() as t:
        y_hat = w * x + b  # 예측y
        loss = (y_hat - y) ** 2  # 손실 오차제곱함수
    grads = t.gradient(loss, [w, b])  # w와 b의 편미분
    w.assign_sub(learning_rate * grads[0])  # 편미분에 학습률을 곱하여 w와 b에서 뺀다
    b.assign_sub(learning_rate * grads[1])


# 지정된 횟수만큼 반복
for i in range(epochs):
    for k in range(len(y)):
        train_step(x[k], y[k])

# 학습된 파라미터
print('w : {:8.5f}   b : {:8.5f}'.format(w.numpy(), b.numpy()))  # 학습결과 y_hat = 0.5x * 1.5


# 학습된 파라미터를 이용한 모델 실행
f = 'x:{:8.5f} --> y:{:8.5f}'
for k in range(len(y)):
    y_hat = w * x[k] + b
    print(f.format(x[k].numpy(), y_hat.numpy()))



# 그래프 실행 모드
train_step_graph = tf.function(train_step)
for i in range(epochs):
    for k in range(len(y)):
        train_step_graph(x[k], y[k])


@tf.function
def train_step_G(x, y):
    with tf.GradientTape() as t:
        y_hat = w * x + b
        loss = (y_hat - y) ** 2
    grads = t.gradient(loss, [w, b])
    w.assign_sub(learning_rate * grads[0])
    b.assign_sub(learning_rate * grads[1])