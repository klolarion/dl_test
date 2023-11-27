import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras.mixed_precision.loss_scale_optimizer

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers, losses, optimizers





def prepare_data():
    iris = load_iris()
    X = iris.data[:, 2:]
    y = iris.target
    lbl_str = iris.target_names
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20)
    return X_tr, y_tr, X_val, y_val, lbl_str


def visualize(net, X, y, multi_class, labels, class_id, colors, xlabel, ylabel, legend_loc='lower right'):
    # 데이터의 최소~최대 범위를 0.05 간격의 좌표값으로 나열
    x_max = np.ceil(np.max(X[:, 0])).astype(int)
    x_min = np.floor(np.min(X[:, 0])).astype(int)
    y_max = np.ceil(np.max(X[:, 1])).astype(int)
    y_min = np.floor(np.min(X[:, 1])).astype(int)
    x_lin = np.linspace(x_min, x_max, (x_max - x_min) * 20 + 1)
    y_lin = np.linspace(y_min, y_max, (y_max - y_min) * 20 + 1)

    # x_lin과 y_lin의 격자좌표의 x와 y값 구하기
    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

    # (x, y) 좌표의 배열로 만들어 신경망의 입력 구성
    X_test = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    # 학습된 신경망으로 X_test에 대한 출력 계산
    if multi_class:
        y_hat = net.preidct(X_test)
        y_hat = np.array([np.argmax(y_hat[k]) for k in range(len(y_hat))], dtype=int)
    else:
        y_hat = (net.predict(X_test) >= 0.5).astype(int)
        y_hat = y_hat.reshape(len(y_hat))

    # 출력할 그래프의 수평/수직 범위 및 각 클래스에 대한 색상 및 범례 설정
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 클래스별 산점도 그리기
    for c, i, c_name in zip(colors, labels, class_id):
        # 격자 좌표의 클래스별 산점도
        plt.scatter(X_test[y_hat == i, 0], X_test[y_hat == i, 1],
                    c=c, s=5, alpha=0.3, edgecolors='none')
        # 학습 표본의 클래스별 산점도
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    c=c, s=20, label=c_name)

    # 범례의 표시 위치 지정
    plt.legend(loc=legend_loc)
    # x축과 y축의 레이블을 지정한 후 그래프 출력
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.show()


nSamples = 150
nDim = 2
nClasses = 3
X_tr, y_tr, X_val, y_val, labels = prepare_data()

# 모델 정의
bp_model_tf = keras.Sequential()
bp_model_tf.add(layers.InputLayer(input_shape=(nDim,)))  # 입력층
bp_model_tf.add(layers.Dense(4, activation='sigmoid'))  # 은닉층
bp_model_tf.add(layers.Dense(nClasses, activation='softmax'))  # 출력층

bp_model_tf.summary()  # 모델의 요약정보 출력




# 모델 훈련을 위한 설정

# 신규버전에서 optimizers 패키지가 변경됨
# tf.python.keras.optimizers.py
# all_classes = {
#       'adadelta': adadelta_v2.Adadelta,
#       'adagrad': adagrad_v2.Adagrad,
#       'adam': adam_v2.Adam,
#       'adamax': adamax_v2.Adamax,
#       'nadam': nadam_v2.Nadam,
#       'rmsprop': rmsprop_v2.RMSprop,
#       'sgd': gradient_descent_v2.SGD,
#       'ftrl': ftrl.Ftrl,
#       'lossscaleoptimizer': loss_scale_optimizer.LossScaleOptimizer,
#       # LossScaleOptimizerV1 deserializes into LossScaleOptimizer, as
#       # LossScaleOptimizerV1 will be removed soon but deserializing it will
#       # still be supported.
#       'lossscaleoptimizerv1': loss_scale_optimizer.LossScaleOptimizer,
#   }

bp_model_tf.compile(optimizer=optimizers.gradient_descent_v2.SGD(0.1, momentum=0.9),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
