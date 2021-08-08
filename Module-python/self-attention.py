# @Auther : wuwuwu 
# @Time : 2021/7/26 
# @File : self-attention.py
# @Description : self-attention 模块造轮子

import numpy as np


class selfAttentionLayer:
    """
    todo list:
    a、 输入矩阵的行列转置，[[a1], [a2], [a3]] --> [batch_size, input_shape]
    b、 W_q, W_k, W_v 组合成一个权重 W = [W_q, W_k, W_v] --> [3, input_shape, batch_size]
    c、 权重尺寸有 bug
    """
    def __init__(self, input_shape):
        """

        :param input_shape: input shape [height, width]
        """
        self.input_shape = input_shape
        self.W_q = np.random.randn(self.input_shape[0], self.input_shape[0])
        self.W_k = np.random.randn(self.input_shape[0], self.input_shape[0])
        self.W_v = np.random.randn(self.input_shape[0], self.input_shape[0])

    def dotProduct(self, a_i, a_j):
        """
        计算第 i 个向量和第 j 个向量之间的注意力关系
        :param a_i: 第 i 个向量
        :param a_j: 第 j 个向量
        :return: 权值 alpha
        """
        assert a_i.shape == a_j.shape
        q = self.W_q.dot(a_i)
        k = self.W_k.dot(a_j)
        return q * k

    def softmax(self, I, axis=0):
        """
        softmax layer
        :param I: Input
        :param axis:
        :return: output
        """
        I = I / I.max()
        I_exp = np.exp(I)
        I_sum = np.sum(I_exp, axis=axis, keepdims=True)
        I_result = I_exp / I_sum
        return I_result

    def forward(self, I):
        """
        self-attention layer
        :param I: Input Matrix
        :return: Output Matrix
        """
        Q = self.W_q.dot(I)
        K = self.W_k.dot(I)
        V = self.W_v.dot(I)
        A = np.dot(K.T, Q) / np.sqrt(self.W_k.shape[0])
        A_prime = self.softmax(A, axis=-1)
        O = np.dot(V, A_prime)
        return O


class multiHeadAttentionLayer:
    def __init__(self, input_shape, head=2):
        """

        :param n: input shape
        :param head: the number of head
        """
        self.input_shape = input_shape
        self.head = head
        self.W_q = np.random.randn(self.input_shape[0], self.input_shape[0], self.head)
        self.W_k = np.random.randn(self.input_shape[0], self.input_shape[0], self.head)
        self.W_v = np.random.randn(self.input_shape[0], self.input_shape[0], self.head)
        self.W_o = np.random.randn(self.input_shape[0], self.input_shape[0] * self.head)

    def softmax(self, I, axis=0):
        """
        softmax layer
        :param I: Input
        :param axis:
        :return: output
        """
        I = I / I.max()
        I_exp = np.exp(I)
        I_sum = np.sum(I_exp, axis=axis, keepdims=True)
        I_result = I_exp / I_sum
        return I_result

    def forward(self, I):
        Q = self.W_q[..., 0].dot(I)
        K = self.W_k[..., 0].dot(I)
        V = self.W_v[..., 0].dot(I)
        A = np.dot(K.T, Q) / np.sqrt(self.W_k.shape[0])
        A_prime = self.softmax(A, axis=-1)
        output = np.dot(V, A_prime)
        for i in range(1, self.head):
            Q = self.W_q[..., i].dot(I)
            K = self.W_k[..., i].dot(I)
            V = self.W_v[..., i].dot(I)
            A = np.dot(K.T, Q) / np.sqrt(self.W_k.shape[0])
            A_prime = self.softmax(A, axis=-1)
            O = np.dot(V, A_prime)
            output = np.concatenate([output, O])
        output = np.dot(self.W_o, output)
        return output


if __name__ == '__main__':
    I = np.arange(12).reshape(4, 3)
    self_Attention = selfAttentionLayer(I.shape)
    Multi_Head_Attention = multiHeadAttentionLayer(I.shape, head=2)
    # O = self_Attention.forward(I)
    O = Multi_Head_Attention.forward(I)
    print(O)
