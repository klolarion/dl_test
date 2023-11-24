import tensorflow as tf
import numpy as np
import keras as ks

# 상수 텐서
a = tf.constant(10.)  # 스칼라
b = tf.constant([1, 2, 3, 4])  # 1차원 벡터
c = tf.constant([
                 [[1, 2], [3, 4], [5, 6]],
                 [[7, 8], [9, 10], [11, 12]],
                 [[13, 14], [15, 16], [17, 18]]], dtype=tf.float32)  # 3차원 행렬
print('a: dtype =', a.dtype, '\n', a)  # shape ()
print('b: shape =', b.shape, '\n', b)  # shape (4, )
print('c: shape =', c.shape, '\n', c)  # shape (3, 3, 2)
print('c: device =', c.device)


# 변수 텐서

x = tf.Variable(10.)
y = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
z = np.array([[1., 3.], [2., 4.], [3., 5.]], dtype=np.float32)
print('x: dtype =', x.dtype, '\n', x)
print('y: shape =', y.shape, '\n', y)  # tensor
print('z: shape =', z.shape, '\n', z)  # numpy array
print('y: device =', y.device)


# 변수텐서는 =, 대입연산자를 사용하면 변수를 수정하는게 아니라 새로운 텐서를 만든다
# 따라서 assign, assign_add, assign_sub 등의 메서드를 사용해야한다.
x.assign_add(20.)  # 기존 x텐서의 값인 10.0에 20.0을 더한 값이 된다
print('x =', x.numpy())  # 30.0


#  tf.math모듈을 이용해서 다양한 수학 함수 사용가능
#  math는 생략하고 사용 가능
print('a * b =', (a * tf.cast(b, tf.float32)).numpy())  # 데이터타입이 다른 경우 변환 후 연산을 진행한다
print('tf.math.exp(y) =', tf.exp(y))
print('tf.math.reduce_sum(c, axis=2) =', tf.reduce_sum(c, axis=2))


# 행렬에 대한 선형대수 연산은 tf.linalg모듈에서 제공하는 함수로 가능
# math와 마찬가지로 생략 가능하다
print('tf.linalg.matmul(x, z) =')
print(tf.matmul(y, z))  # 행렬곱 연산 tf반환
print('np.matmul(y, z = ')
print(np.matmul(y, z))  # 행렬곱 연산 numpy반환

