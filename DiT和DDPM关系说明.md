# DiT 和 DDPM 的关系

## 一句话理解

DDPM 是扩散模型的“训练和采样框架”，DiT 是这个框架里面的“去噪网络”。

在普通 DDPM 里，去噪网络常用 UNet：

```text
epsilon_theta = UNet(x_t, t)
```

在 DiT 里，只是把 UNet 换成 Transformer：

```text
epsilon_theta = DiT(x_t, t)
```

所以 DiT 不是把 DDPM 公式推翻重写，而是把 DDPM 里面负责预测噪声的神经网络换成了 Vision Transformer 风格的结构。

## DDPM 负责什么

DDPM 负责两条过程。

第一条是训练时的前向加噪：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

这里 `x_0` 是原图，`epsilon` 是真实随机噪声，`x_t` 是第 `t` 步的带噪图片。

第二条是采样时的反向去噪：

```text
x_T -> x_{T-1} -> ... -> x_0
```

每一步都要让模型预测当前噪声：

```text
epsilon_theta(x_t, t)
```

然后用 DDPM 的反推公式算出 `x_{t-1}`。

## DiT 负责什么

DiT 只负责学习这个函数：

```text
epsilon_theta(x_t, t)
```

也就是说，DiT 的输入是：

```text
带噪图片 x_t
时间步 t
```

DiT 的输出是：

```text
预测噪声 epsilon_theta
```

训练 loss 仍然是 DDPM 常用的噪声预测 MSE：

```text
loss = MSE(epsilon_theta(x_t, t), epsilon)
```

## DiT 为什么要 patchify

Transformer 处理的是 token 序列，不是直接处理 `[B, C, H, W]` 的图片张量。

所以 DiT 先把图片切成 patch：

```text
[B, C, H, W] -> [B, N, hidden_size]
```

其中：

- `B` 是 batch size
- `N` 是 patch 数量
- `hidden_size` 是每个 patch token 的 embedding 维度

Transformer block 处理完 token 后，再把 token 还原回图片形状：

```text
[B, N, patch_size * patch_size * C] -> [B, C, H, W]
```

这个还原过程叫 `unpatchify`。

## 时间步 t 怎么进入 DiT

DDPM 必须告诉模型当前噪声强度，所以 `t` 很重要。

DiT 先把 `t` 做成 timestep embedding，然后用 AdaLN 的方式调制每个 Transformer block：

```text
LayerNorm(x) * (1 + scale(t)) + shift(t)
```

直观理解：不同时间步的噪声强度不同，模型在不同 `t` 下应该采用不同的去噪策略。

## 当前 notebook 的结构

`从零开始手搓DiT diffusion model.ipynb` 里面分成两部分：

- DiT 部分：`PatchEmbed`、`DiTBlock`、`FinalLayer`、`DiT`
- DDPM 部分：`GaussianDiffusionTrainer`、`GaussianDiffusionSampler`

重点看这行：

```python
pred_noise = self.model(x_t, t)
```

这里的 `self.model` 可以是 UNet，也可以是 DiT。换成 DiT 后，DDPM 的加噪公式、loss、采样公式都不需要变。
