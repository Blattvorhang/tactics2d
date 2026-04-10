# 高速场景跟车算法设计与实现

## 1. 问题描述

本实验基于 NGSIM 真实轨迹数据，实现一个跟车算法，使自车能够在高速场景中稳定跟随目标车辆。

评价指标主要包括：

1. **跟车距离稳定性**：自车与目标车辆间距保持在 10–20 m；
2. **转向平滑性**：方向盘转角变化满足 $\Delta \theta < 0.2 \text{rad}$。

## 2. 算法总体设计

本算法采用**纵向与横向解耦控制结构**：

* **纵向控制（Longitudinal Control）**：基于
  → Intelligent Driver Model（IDM）

* **横向控制（Lateral Control）**：基于几何方法
  → Pure Pursuit

* **平滑模块（Smoothing）**：用于抑制控制抖动

整体结构如下：

```text
Target Vehicle → Relative State → Controller → Action (steering, accel)
                             ↓
              Longitudinal (IDM) + Lateral (Pure Pursuit)
                             ↓
                     Smoothing & Constraints
```

## 3. 状态建模

系统输入包括：

* 自车状态：

  * 位置：$(x_{ego}, y_{ego})$
  * 航向角：$\theta_{ego}$
  * 速度：$v_{ego}$

* 目标车状态：

  * 位置：$(x_{lead}, y_{lead})$
  * 速度：$v_{lead}$

定义相对量：

$$
dx = x_{lead} - x_{ego}, \quad dy = y_{lead} - y_{ego}
$$

沿车头方向的有效距离：

$$
d = dx \cos\theta + dy \sin\theta
$$

相对速度：

$$
\Delta v = v_{ego} - v_{lead}
$$

## 4. 纵向控制（IDM）

采用 Intelligent Driver Model 计算加速度：

### 4.1 安全距离

$$
s^* = s_0 + vT + \frac{v \Delta v}{2\sqrt{ab}}
$$

其中：

* $s_0$：最小间距
* $T$：期望时距
* $a$：最大加速度
* $b$：舒适减速度

### 4.2 加速度控制律

$$
a = a_{max} \left(1 - (v/v_0)^\delta - (s^*/d)^2 \right)
$$

### 4.3 特点

* 自动适应不同速度下的安全距离
* 避免碰撞（collision-free）
* 抑制跟车振荡（improved stability）

## 5. 横向控制（Pure Pursuit）

### 5.1 基本思想

通过跟踪前方一个“目标点”来计算转向，使车辆平滑逼近目标轨迹。

### 5.2 前视距离

$$
L_d = \max(k \cdot v, L_{min})
$$

### 5.3 曲率计算

将目标点转换到自车坐标系：

$$
\kappa = \frac{2y}{L_d^2}
$$

转向角：

$$
\theta = \arctan(\kappa)
$$

### 5.4 特点

* 几何方法，计算简单
* 对噪声不敏感
* 自带低通特性（抑制抖动）


## 6. 控制平滑与约束

为满足转向平滑性指标，设计如下机制：


### 6.1 低通滤波（Low-pass Filter）

$$
\theta_t = \alpha \theta_{t-1} + (1-\alpha)\theta_{raw}
$$

作用：

* 消除高频噪声
* 平滑控制输出

### 6.2 转向变化率限制（Rate Limiter）

$$
\Delta \theta = \text{clip}(\theta_t - \theta_{t-1}, -\delta, \delta)
$$

保证：

$$
|\Delta \theta| < 0.2 , \text{rad}
$$

### 6.3 加速度约束

$$
a \in [-4, 2] , \text{m/s}^2
$$

## 7. 方法优势分析

### 7.1 满足实验指标

| 指标             | 设计机制       |
| -------------- | ---------- |
| 距离稳定（10–20m）   | IDM 动态安全距离 |
| 转向平滑（∆θ < 0.2） | 低通滤波 + 限幅  |

### 7.2 相比传统方法优势

相比简单 PID 跟车：

* IDM 提供**非线性稳定控制**
* Pure Pursuit 避免**角度振荡问题**
* 解耦结构提高鲁棒性

### 7.3 工程合理性

该方法具有以下工程特性：

* 计算量低（适合实时控制）
* 参数具有物理意义（易调参）
* 在真实轨迹数据（NGSIM）上表现稳定