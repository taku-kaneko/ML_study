# 第 8 章 大数の法則と中心極限定理（中井検裕）

この章はとんでもなく難しいのでさらっといく．

## 大数の法則

### 概要とシミュレーション

大数の法則は，標本の大きさ$n$が十分に大きければ，標本平均$\bar{X} = (X_1 + X_2 + \cdots + X_n) / n$の確率分布は本来の確率分布（母集団確率分布）の平均（母平均）$\mu$の近くに集中していることを保証するものである．

コインを 10 回投げるとして，表か裏が出る確率を 1/2 とする．$i$回目に表が出た場合に 1，裏が出た場合に 0 をとる確率変数$x_i$を考える．表が出た頻度$r$は確率変数の和

$$
r = x_1 + x_2 + \cdots + x_n
$$

である．
表が出た回数の割合$\hat{p} = r/10$は観測された成功率であって，観測された成功回数によって$\hat{p} = 0, 0.1, 0.2,\cdots$となる．一般に，$m$をコイン投げの回数とするとき，$r/n$は相対度数である．$r$は確率変数で，$n=10, p=0.5$の二項分布$Bi(10, 0.5)$，すなわち

$$
f_{10}(x) = {}_{10}\mathrm{C}_{x} (1/2)^{10}, \quad x=0, 1, 2, \cdots, 10
$$

に従い，その期待値，分散は

$$
\mathbb{E}(r) = np = 5, \quad V(r) = np(1 - p) = 2.5
$$

だから，割合$r/n$の期待値，分散は

$$
\mathbb{E}(r/n) = p = 0.5, \quad V(r/n) = p(1 - p) / n = 0.025
$$

であり，$p=0.5$は真の成功率となっている．
成功の割合が$x/10$となる確率を$f_{10}(x)$で実際に計算すると以下のようになり，
期待値である真の確率$0.5$およびその周辺の発生確率が高いが，$0.2$以下および$0.8$以上の確率も 11%近くある．

ここで，コイン投げの回数$n$を増やしたときに，期待値$p = 0.5$周辺の確率がどのように変化していくか見てみる．
今回は，$n=10, 20, 30, 40, 50, 100$としたときの，$P(0.4 \leq r/n \leq 0.6)$の変化を見てみる．

結果は以下の通りで，$n$が大きくなるにつれて観測された成功率$\hat{p}=r/n$が 0.4 から 0.6 までの確率は高くなり，$n=100$では$96\%$を超えてほとんどの値が真の成功率$p=0.5$の周囲に集中する．
![](https://drive.google.com/uc?export=view&id=1-daYK1K1A6brg3ZNubbj-VGlRn3jm5cb)

もう一つシミュレーションをする．成功の確率を$p=0.4$としたとき，ベルヌーイ試行を 2 万回行ってみる．
このとき，100 回おきに区切り，1 回目からの観測された成功率を計算すれば，大数の法則により，この成功率は試行回数$n$が増えるたびに真の成功率に近づくはず．
この実験を 4 回行い，試行回数に対する観測された成功率の推移をみる．

結果は以下の通りで，試行回数が増えるたびに真の成功確率に近づいていることがわかる．
また，すべての実験が真の成功率に収束しているわけではないこともわかる．
これは，大数の法則の意味は，試行回数$n$が大きくなれば観測された成功率は真の成功率の近傍にある確率が極めて高くなるといくことであり，時々起こる特定の場合には，そうなっていないこともあるという例である．

![](https://drive.google.com/uc?export=view&id=1iNiaDZgl1Ak8Pgpu5OBDofg_xzZNrTYD)

<details>
<summary>シミュレーションに使用したコード</summary>
一つ目のグラフ

```Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def Bi(x, n, p):
    res = comb(n, x) \* (p) \*\* n
    return res

n_list = [10, 20, 30, 40, 50, 100]
x_list = []
prob_list = []
for n in n_list:
    x = np.arange(0, n + 1, 1)
    prob = Bi(x, n, 0.5)
    x_list += [x]
    prob_list += [prob]

p_list = []
for x, prob, n in zip(x_list, prob_list, n_list):
    ind = np.where((x / n >= 0.4) & (x / n <= 0.6))[0]
    p = prob[ind].sum()
    p_list += [p]

fig, ax = plt.subplots(figsize=[4, 4])
ax.plot(n_list, p_list)
ax.set_xticks(n_list)
ax.set_xticklabels(n_list)
ax.set(xlabel=r"$n$", ylabel=r"$P(0.4 \leq r/n \leq 0.6)$")

```

二つ目のグラフ

```Python
exam = 4
p = 0.4
N = 20000
r_list = []
for i in range(exam):
    u = np.random.rand(N)
    n_list = np.arange(1, N + 100, 100)
    r = [(u[:n] < p).sum() / n for n in n_list]
    r_list += [r]
r_list = np.array(r_list)

fig, ax = plt.subplots(figsize=[4, 4])
ax.plot(n_list, r_list.T)
ax.axhline(y=p, ls="dashed", c="k")
ax.set_ylim(0.35, 0.45)
ax.set(xlabel="Number of trials", ylabel=r"$p=r/n$", title=r"$p=0.4$")
```

</details>

### 数学的にきちんと証明する

大数の法則は正式には２種類あり，今回証明するのは簡単な大数の弱法則である．
大数の弱法則は，平均$\mu$，分散$\sigma^2$の分布に従う互いに独立な確率変数$X_1, X_2, \cdots$と任意の$\varepsilon \gt 0$に対して

$$
\lim_{n\rightarrow\infty}P\left(\left|\frac{X_1 + X_2 + \cdots + X_n}{n} - \mu\right| \geq \varepsilon \right) = 0
$$

となることである．
式を見れば，サンプルの平均と真の平均の差が$\varepsilon$より大きくなる確率が$n\rightarrow\infty$で 0 になることを示している．
つまり，サンプルの平均と真の平均はサンプル数が無限大になれば限りなく近くなることを示している．

大数の弱法則の証明の流れは

1. マルコフの不等式の証明
1. チェビシェフの不等式の証明
1. 大数の弱法則

という流れ．今回はチェビシェフの不等式が成り立つことを前提に進んでいく．

標本サンプル

$$
Y_n = \frac{X_1 + X_2 + \cdots + X_n}{n}
$$

とおくと，その期待値と分散は

$$
\mathbb{E}(Y_n) = \frac{n\mu}{n} = \mu, \quad V(Y_n) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

となる．
よってチェビシェフの不等式より

$$
P(|Y_n - \mu| \geq \varepsilon) \leq \frac{\sigma^2}{n\varepsilon^2}
$$

となり，両辺で$n \rightarrow \infty$の極限を取れば

$$
\lim_{n\rightarrow\infty}P(|Y_n - \mu| \geq \varepsilon) \leq 0
$$

となり，大数の弱法則は示された．ちなみにこれを確率収束というらしい．

ちなみに大数の強法則は証明がかなり難しいらしい．
また違いは微妙らしい．

## 中心極限定理

中心極限定理は，母集団分布が何であっても，和$X_1 + X_2 + \cdots + X_n$の確率分布の形は，$n$が大きい時には大略正規分布と考えてよいということを数学的に保証しているものである．

母集団分布の平均と分散を$\mu, \sigma^2$とするとき，母集団分布が何であっても標本の大きさ$n$が大きいときは，大略

$$
\begin{aligned}
S_n &= X_1 + X_2 + \cdots + X_n \Rightarrow \mathcal{N}(n\mu, n\sigma^2) \\
\bar{X} &= (X_1 + X_2 + \cdots + X_n) / n \Rightarrow \mathcal{N}(\mu, \sigma^2/n)
\end{aligned}
$$

に従うと考えてよい．

サイコロの例に中心極限定理を実感として把握してみる．
サイコロの目の出方は離散型の一様分布であり，$\mu=7/2, \sigma^2=35/12$であり，正規分布とは相当に違う．
試しにシミュレーションしてみると以下のようになり，正規分布に従うとは到底いえない形をしている．

![](https://drive.google.com/uc?export=view&id=1F36Sb5oQwcaHL6Mgqp5Bbjvj7NlUmdJ1)

一方で，二つのサイコロの目の和の平均$(X_1 + X_2) / 2$の分布は以下のようになり，正規分布のような形になっている．

![](https://drive.google.com/uc?export=view&id=16DXcjaIxqwxlF-idO8MhyZ8kNVQPrr6d)

一様分布のような左右対象な分布の場合は，正規分布への近似速度が早いため$n$が小さくても正規分布に近い形状をとる．
例えば，以下のような指数分布の場合は一様分布のときよりも正規分布への近似速度は遅いが，指数分布に従う確率変数の標本平均は，$n$が大きければ中心極限定理によって正規分布に近づいていく．

指数分布からサンプルした確率変数の分布（1000 個）
![](https://drive.google.com/uc?export=view&id=1d0b976ZjyFzjCsUcw4OOM_5rguqWDIds)

指数分布からサンプルした 1000 個の確率変数の平均をとったものの分布（1000 個）
![](https://drive.google.com/uc?export=view&id=1F0JQTqKinEQVghbBCfNRGKwSMAw-EffK)

<details>
<summary>シミュレーションに使用したコード</summary>
一つ目のグラフ（一様分布）

```Python
p = np.random.randint(1, 7, 1000)

x = np.unique(p, return_counts=True)[0]
y = np.unique(p, return_counts=True)[1] / p.size

fig, ax = plt.subplots(figsize=[4, 4])
ax.bar(x, y)
```

二つ目のグラフ（一様分布からサンプルした確率変数の平均の分布）

```Python
p = np.array([np.mean(np.random.randint(1, 7, 2)) for i in range(20000)])

x = np.unique(p, return_counts=True)[0]
y = np.unique(p, return_counts=True)[1] / p.size

fig, ax = plt.subplots(figsize=[4, 4])
ax.bar(x, y)
```

三つ目のグラフ（指数分布）

```Python
p = np.random.exponential(1 / 0.1, 1000)

fig, ax = plt.subplots(figsize=[4, 4])
weights = np.ones_like(p) / p.size
ax.hist(p, weights=weights)
```

４つ目のグラフ（指数分布からサンプルした確率変数の平均の分布）

```Python
p = np.array([np.mean(np.random.exponential(1 / 0.1, 1000)) for i in range(1000)])

fig, ax = plt.subplots(figsize=[4, 4])
weights = np.ones_like(p) / p.size
ax.hist(p, weights=weights)
```

</details>

### 数学的にきちんと証明する

中心極限定理をきちんと述べると，次の条件

1. $X_1, X_2, \cdots, X_n$が互いに独立で同じ分布に従う
1. $\mathbb{E}(X_i) = \mu, \quad V(X_i) = \sigma^2\quad (i=1, 2, \cdots, n)$

の下で確率変数$Y = (\sqrt{n}(\bar{X} - \mu)) / \sigma$は$\mathcal{N}(0, 1)$に弱収束する．
つまり，

$$
\lim_{n\rightarrow\infty}P\left(\frac{\sqrt{n}(\bar{X} - \mu)}{\sigma} \leq y\right) = \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}} \mathrm{d} x
$$

である．

この証明は$Y_i$のモーメント母関数が$\mathcal{N}(0, 1)$のモーメント母関数に近づくことを示せばよい．

## 中心極限定理の応用

### 二項分布を正規分布で近似する

二項分布$Bi(n, p)$は

$$
f(x) = {}_n\mathrm{C}_x p^x (1 - p)^{n-x}
$$

で計算できるが，$n$が大きい場合に${}_n\mathrm{C}_x$の計算が困難になる．例えば，$n=4000$のときは，$4000!$に近い計算をしなければならないが，これを計算するのは困難だと容易に想像できる．
こんなときは，中心極限定理により正規分布に近似することで計算を容易にできる．

**手順**

$n$回の試行のうち，成功の回数$S$が$k$回以上$k'$回以下である確率を中心極限定理を用いて算出する．

1. 二項分布における成功の回数$S = X_1 + X_2 + \cdots + X_n$を用意する．ここで，$X_i$は$Bi(1, p)$に従う確率変数である．
1. $\mathbb{E}(S) = np, \quad V(S) = np(1 - p)$を用いて標準化変数$$z = \frac{S - \mu}{\sigma} = \frac{S - np}{\sqrt{np(1 - p)}}$$を用意する．$z$の確率分布は$n$が大きいときは標準正規分布$\mathcal{N}(0, 1)$によって近似される．
1. 標準正規分布の累積分布関数$\Phi$により成功の回数$S$が$k$回以上$k'$回以下である確率は，$$
\begin{aligned}
P(k \leq S \leq k') &= P\left(\frac{k - np}{\sqrt{np(1-p)}} \leq z \leq \frac{k' - np}{\sqrt{np(1-p)}} \right) \\
&\fallingdotseq \Phi\left(\frac{k' - np}{\sqrt{np(1-p)}}\right) - \Phi\left(\frac{k - np}{\sqrt{np(1-p)}}\right)
\end{aligned}$$で求めることができる．
1. あとは正規分布表を参考に確率を計算すれば OK

$n$がどのくらい大きければ正規分布による近似を用いてよいかについては，$np \gt 5$かつ
$n(1-p) \gt 5$を満たせれば OK らしい．
$p$が 0 もしくは 1 に近い場合は$n$が相当大きくなければならなく，$p$が$1/2$に近ければ$n$は 10 とかでよい．
これは，中心極限定理の収束速度は元の分布が左右対象であるほど早く，左右非対称ほど遅くなることに起因している．

### 正規乱数を発生させる

区間$(0, 1)$上の一様乱数（一様分布に従う乱数）を中心極限定理を用いて任意の$\mu, \sigma^2$を持つ正規分布に従う乱数に変換することができる．

**手順**

1. 区間$(0, 1)$上の一様乱数を$n$個発生させる．これを$r_1, r_2, \dots, r_n$とする．
1. 乱数の和$S = \sum_{i=1}^n r_i$を作る．
1. $\mathbb{E}(S) = \mu_S =  n(1 + 0) / 2 = n / 2, \quad V(S) = \sigma_S^2 = n(1 - 0)^2 / 12 = n/12$を用いて標準化変数$$z=\frac{S - \mu_S}{\sigma_S} = \frac{S - n/2}{\sqrt{n/12}}$$を作る．この$z$は中心極限定理により$n$が大きいとき$\mathcal{N}(0, 1)$に従う乱数である．
1. 標準化の逆変換$x = \sigma z + \mu$を用いて，$\mathcal{N}(0, 1) \rightarrow \mathcal{N}(\mu, \sigma^2)$に変換する．
