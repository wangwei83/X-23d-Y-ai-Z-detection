根据上文中描述的非线性最小二乘法问题和代码实现，所求解的模型可以写成以下的数学公式：

### 模型函数

假设我们的模型函数 \( f \) 形式如下：

\[
f(x, a, b) = a \cdot e^{bx}
\]

其中：
- \( x \) 是自变量。
- \( a \) 和 \( b \) 是需要通过数据拟合来确定的参数。

### 目标函数

在最小二乘法中，我们的目标是最小化观测数据与模型预测之间的残差平方和。具体来说，如果有一组观测数据 \((x_i, y_i)\) ，我们的目标是最小化以下目标函数：

\[
\sum_{i=1}^{n} \left( y_i - f(x_i, a, b) \right)^2
\]

### 参数估计

使用非线性最小二乘法，我们寻找参数 \( a \) 和 \( b \)，使得目标函数达到最小值。这通常通过迭代算法实现，如 Levenberg-Marquardt 算法，这是 `curve_fit` 函数背后的算法之一。

### 结果表示

假设通过最小二乘法得到的最优参数估计值为 \( \hat{a} \) 和 \( \hat{b} \)，则拟合得到的模型可以表示为：

\[
f(x, \hat{a}, \hat{b}) = \hat{a} \cdot e^{\hat{b}x}
\]

这是拟合得到的非线性模型的数学表示，可以用来预测或解释数据。