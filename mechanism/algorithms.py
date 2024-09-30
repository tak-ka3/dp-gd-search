import torch
import numpy as np
import torch.nn as nn
from enum import Enum
from abc import ABC, abstractmethod

class InputPattern(Enum):
    ALL_ONE_DIFF = 1
    ONE_ONE_DIFF = 2

class OutputType(Enum):
    DISC = 1
    CONT = 2

def softargmax1d(input, beta=10):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1) # 大きい値を持つインデックスに1がたつ
    indices = torch.linspace(0, 1, n) # [0, 0.25, 0.50, 0.75, 1.00]
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result

class Alg(ABC):
    def __init__(self, eps):
        self.eps = eps
    @abstractmethod
    def mech(self, x):
        pass
    @abstractmethod
    def get_eps(self, x, n_samples: int = 1):
        pass

class NoisySum(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.CONT
    def mech(self, x, n_samples: int = 1):
        return torch.sum(x) + torch.distributions.laplace.Laplace(0, 1 * len(x) / self.eps).sample()
    def get_eps(self, x, n_samples: int = 1):
        return self.eps * n_samples

class NoisyMax(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.CONT
    def mech(self, x, n_samples: int = 1):
        noisy_vec = []
        for i in range(len(x)):
            noisy_vec.append(x[i] + torch.distributions.laplace.Laplace(0, 1 / (self.eps/2)).sample())
        return torch.max(torch.stack(noisy_vec))
    def get_eps(self, x, n_samples: int = 1):
        return self.eps/2*len(x) * n_samples

class NoisyArgMax(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, x, n_samples: int = 1):
        noisy_vec = []
        for i in range(len(x)):
            noisy_vec.append(x[i] + torch.distributions.laplace.Laplace(0, 1 / (self.eps/2)).sample())
        return softargmax1d(torch.stack(noisy_vec))
    def get_eps(self, x, n_samples: int = 1):
        return self.eps * n_samples


class NoisyMaxExp(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.CONT
    def mech(self, x, n_samples: int = 1):
        noisy_vec = []
        for i in range(len(x)):
            noisy_vec.append(x[i] + torch.distributions.exponential.Exponential(self.eps/2).sample())
        return torch.max(torch.stack(noisy_vec))
    def get_eps(self, x, n_samples: int = 1):
        return self.eps/2*len(x) * n_samples

class NoisyArgMaxExp(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, x, n_samples: int = 1):
        noisy_vec = []
        for i in range(len(x)):
            noisy_vec.append(x[i] + torch.distributions.exponential.Exponential(self.eps/2).sample())
        return softargmax1d(torch.stack(noisy_vec))
    def get_eps(self, x, n_samples: int = 1):
        return self.eps * n_samples

# 1つの要素だけが最大で1変わるような場合を想定
class NoisyHist1(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ONE_ONE_DIFF
        self.output_type = OutputType.CONT
    def mech(self, x, n_samples: int = 1):
        l = x.shape[0]
        v = np.atleast_2d(x)

        m = v + np.random.laplace(scale=1/self.eps, size=(n_samples, l))
        return m
    def get_eps(self, x, n_samples: int = 1):
        return self.eps * n_samples
    
# wrong implementation
class NoisyHist2(Alg):
    def __init__(self, eps):
        self.eps = eps
        self.input_pattern = InputPattern.ONE_ONE_DIFF
        self.output_type = OutputType.CONT
    def mech(self, x, n_samples: int = 1):
        l = x.shape[0]
        v = np.atleast_2d(x)

        m = v + np.random.laplace(scale=self.eps, size=(n_samples, l))
        return m
    def get_eps(self, x, n_samples: int = 1):
        return 1 / self.eps * n_samples
    
class SVT1(Alg):
    """
    Alg. 1 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, a, n_samples: int = 1):
        """
        TODO: n_samples = 1を想定した実装となっている
        """
        x = torch.atleast_2d(a)

        rho = torch.distributions.laplace.Laplace(0, 1 / self.eps1).sample((n_samples, 1))
        nu = torch.distributions.laplace.Laplace(0, 1 / self.eps2).sample((n_samples, x.shape[0]))

        alpha = 0.1
        m = nu + x  # broadcasts x_2d vertically
        cmp = torch.sigmoid(alpha * (m - (rho + self.t)))
        count = torch.tensor(0)
        aborted = torch.tensor(0)

        result = []
        for i in range(len(cmp[0])):
            # ブールインデックスを使っており、res[False, ]は最初の軸について、選択する要素がないことを表し空配列を返す
            result.append(aborted * (-1) + (1 - aborted) * cmp[0, i])
            count = count + cmp[0][i] # もしcmp=1ならcountを1増やす
            # 一回aborted=trueになったら、それ以降はaborted=trueになる
            # count == self.cになるまではaborted=falseのまま
            cnt_eq = torch.exp(-10000 * (count - self.c)**2)
            aborted = aborted + cnt_eq - aborted * cnt_eq
        return torch.stack(result)
    def get_eps(self, x, n_samples: int = 1):
        return self.eps * n_samples
    
# Understanding the Sparse Vector Technique for Differential Privacy. VLDB 2017 Alg4
class SVT2(Alg):
    def __init__(self, eps, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 4.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, x, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
                -1 = ABORTED;
        """
        # columns: queries
        # rows: samples
        x_2d = np.atleast_2d(x)
        n_queries = x.shape[0]

        rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
        nu = np.random.laplace(scale=2*self.c / self.eps2, size=(n_samples, n_queries))

        m = nu + x_2d  # broadcasts x vertically

        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = np.empty(shape=m.shape, dtype=int)
        for col_idx in range(0, n_queries):
            cmp = m[:, col_idx] >= (rho + self.t)
            res[:, col_idx] = cmp.astype(int)
            res[aborted, col_idx] = -1
            count = count + cmp

            # update rho whenever we answer TRUE
            new_rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
            rho[cmp] = new_rho[cmp]

            aborted = np.logical_or(aborted, count == self.c)
        return res
    def get_eps(self, x, n_samples: int = 1):
        return self.eps * n_samples
    
# Understanding the Sparse Vector Technique for Differential Privacy. VLDB 2017 Alg3
# 連続的な値と離散的なFalseやAbortedを表す値を返す <- これだけ他と異なる
class SVT3(Alg):
    def __init__(self, eps, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.CONT
    def mech(self, x, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            float ndarray of shape (n_samples, a.shape[0]) with special entries
                -1000.0 = FALSE;
                -2000.0 = ABORTED;
        """
        # columns: queries
        # rows: samples
        x_2d = np.atleast_2d(x)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=self.c / self.eps2, size=(n_samples, x.shape[0]))

        m = nu + x_2d  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)

        res = m
        res[np.logical_not(cmp)] = -1000.0

        col_idx = 0
        for column in cmp.T:
            res[aborted, col_idx] = -2000.0
            count = count + column
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        return res
    def get_eps(self, x, n_samples: int = 1):
        return np.inf
    
class SVT4(Alg):
    def __init__(self, eps, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 4.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, x, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
                -1 = ABORTED;
        """
        # columns: queries
        # rows: samples
        x_2d = np.atleast_2d(x)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=1 / self.eps2, size=(n_samples, x.shape[0]))

        m = nu + x_2d  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = cmp.astype(int)

        col_idx = 0
        for column in cmp.T:
            res[aborted, col_idx] = -1
            count = count + column
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        return res
    def get_eps(self, x, n_samples: int = 1):
        return (1 + 6 * self.c) / 4 * self.eps * n_samples
    
# Understanding the Sparse Vector Technique for Differential Privacy. VLDB 2017 Alg5
class SVT5(Alg):
    def __init__(self, eps: int = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, x, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
        """
        # columns: queries
        # rows: samples
        x_2d = np.atleast_2d(x)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))

        cmp = x_2d >= (rho + self.t)   # broadcasts rho horizontally, x vertically
        return cmp.astype(int)
    def get_eps(self, x, n_samples: int = 1):
        return np.inf
    
# Understanding the Sparse Vector Technique for Differential Privacy. VLDB 2017 Alg6
class SVT6(Alg):
    def __init__(self, eps: float = 0.1, c: int = 2, t: float =1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t
        self.input_pattern = InputPattern.ALL_ONE_DIFF
        self.output_type = OutputType.DISC
    def mech(self, x, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
        """
        # columns: queries
        # rows: samples
        x_2d = np.atleast_2d(x)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=1 / self.eps2, size=(n_samples, x.shape[0]))

        m = nu + x_2d  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        return cmp.astype(int)
    def get_eps(self, x, n_samples: int = 1):
        return np.inf
    
