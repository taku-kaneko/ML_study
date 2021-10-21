# 第 7 章 多次元の確率分布（松原 望）

1 周目は全体の雰囲気を理解するまでに止める

## 同時確率分布と周辺確率分布

### 同時確率分布

二つの確率変数$X, Y$があるとし，これを 2 次元ベクトル$(X,Y)$として表す．とりあえず離散型の確率変数とすると$X=x$であり，かつ$Y=y$である確率

$$
P(X=x, Y=y) = f(x, y)
$$

を 2 次元確率変数$(X, Y)$の同時確率分布という．もちろん

$$
f(x, y) \geq 0, \quad \sum_x\sum_y f(x, y)=1
$$

を満たさなければならない．

$X, Y$が連続型の確率変数の場合は$f(x, y)$は２次元の確率密度関数で同時確率密度関数と呼ばれ，

$$
f(x, y) \geq 0, \quad \iint_S f(x, y) \mathrm{d}x\mathrm{d}y = 1
$$

を満たす．ここで$S$は標本空間である．

### 周辺確率分布

同時確率分布から片方の確率変数を消去した確率分布を周辺分布と呼ぶ．例えば，確率変数$X, Y$のうち，$Y$を以下のように消去したとする．

$$
g(x) = \sum_y f(x, y)
$$

これを，$X$の周辺確率分布と呼ぶ．連続型の場合も同様に

$$
g(x) = \int_{-\infty}^\infty f(x, y) \mathrm{d}y, \quad h(y) = \int_{-\infty}^\infty f(x, y) \mathrm{d}x
$$

を$X, Y$の周辺確率密度関数と呼ぶ．機械学習の文脈では積分によって確率変数を消去することから，積分消去と呼ぶこともある．

### 共分散と相関係数

第 3 章あたりで述べたときと同じように，確率変数でも共分散と相関係数を考えることができる．

確率変数の和の分散は，2 変数$X, Y$の間に関連があれば一方の変化は他方に影響を及ぼすことから，加法が成立しないことが想像できる．実際に

$$
V(X+Y) \neq V(X) + V(Y)
$$

である．
正しくは，$V(X) = \mathbb{E}\{(X-\mu)^2\}$を用いて

$$
\begin{aligned}
V(X+Y) &= \mathbb{E}\left[\left\{(X+Y) - (\mu_X+\mu_Y)\right\}^2\right] \\
&= \mathbb{E}\left[\left\{(X - \mu_X)^2 + (Y - \mu_Y)^2 - 2\mathbb{E}(X - \mu_X)(Y - \mu_Y)\right\}\right] \\
&= V(X) + V(Y) + 2\mathbb{E}\left[(X - \mu_X)(Y - \mu_Y)\right]\\
& = V(X) + V(Y) + 2\mathrm{Cov}(X, Y)
\end{aligned}
$$

となる．ただし，

$$
\mathrm{Cov}(X, Y) = \mathbb{E}\{(X - \mu_X)(Y - \mu_Y)\}, \quad \mu_X = \mathbb{E}(X), \quad \mu_Y = \mathbb{E}(Y)
$$

である．$\mathrm{Cov}(X, Y) \gt 0$なら$X, Y$の大小が同傾向，$\lt 0$なら反対傾向となる．

これを株式投資の話に例えると，A 石油会社と B 石油会社の株式のように同一業種の株式に同時に投資をするのは一般的に勧められない．なぜなら，エネルギー危機などの共通の経済的要因によって A も B も同傾向に連動するから，$\mathrm{Cov}(X, Y) \gt 0$となり，単独の分散の和$V(X) + V(Y)$以上にばらつくからである．つまり，変動のリスクが連動の分だけ大きくなる．

共分散は傾向は分かってもその強さの程度はわからない．そこで登場するのが確率変数$X, Y$の相関係数

$$
\rho_{XY} = \frac{\mathrm{Cov}(X, Y)}{\sqrt{V(X)}\sqrt{(V(Y))}}
$$

である．$\rho_{XY}$は必ず$-1 \leq \rho_{XY} \leq 1$の範囲にはいるので，関連の絶対的な程度を把握することができる．

相関係数の算出に必要な共分散は

$$
\mathrm{Cov}(X, Y) = \mathbb{E}(XY) - \mathbb{E}(X)\mathbb{E}(Y)
$$

で計算されることが多い．なお，共分散は同時確率分布で

$$
\begin{aligned}
\mathrm{Cov}(X, Y) &= \sum_x\sum_y(x-\mu_X)(y-\mu_Y)\cdot f(x, y)\quad \mathrm{(離散型)} \\
\mathrm{Cov}(X, Y) &= \iint_S (x-\mu_X)(y-\mu_Y)\cdot f(x, y) \mathrm{d}x\mathrm{d}y\quad \mathrm{(連続型)}
\end{aligned}
$$

と表されるが，期待値から導出する式を用いて

$$
\begin{aligned}
\mathbb{E}(XY) &= \sum_x\sum_y xy\cdot f(x, y) \quad \mathrm{(離散型)} \\
\mathbb{E}(XY) &= \iint_S xy\cdot f(x, y) \mathrm{d}x\mathrm{d}y \quad \mathrm{(連続型)}
\end{aligned}
$$

から計算する．

## 条件付き確率分布

相関係数$\rho$は$X, Y$の関連の全体的な傾向を見る指標だったけど，元の情報は同時確率分布に含まれている．これを**条件付き確率**としてみてみる．

まずは，条件付き確率の定義を思い出す．

$$
\begin{aligned}
P(X=x | Y=y) &= \frac{P(X=x, Y=y)}{P(Y=y)} \\
P(Y=y | X=x) &= \frac{P(X=x, Y=y)}{P(X=x)}
\end{aligned}
$$

上式は確率であるが，確率分布でも同じようにかけるのでは？ということで書いてみると，

$$
\begin{aligned}
g(x|y) &= \frac{f(x, y)}{h(y)} \\
h(y|x) &= \frac{f(x, y)}{g(x)}
\end{aligned}
$$

となる．これを離散型の場合は条件付き確率分布，連続型の場合は条件付き確率密度関数と呼ぶ．ただし，$h(y) \neq 0, g(x) \neq 0$である．

もちろん，

$$
\sum_x g(x|y) = \sum_x \frac{f(x, y)}{h(y)} = \frac{h(y)}{h(y)} = 1
$$

となり，$g(x|y)$は$x$の関数として確率分布の条件を満たす．

条件付き確率分布も一つの確率分布なので，期待値や分散などを考えることができる．しかも条件付きで．

条件付き期待値と条件付き分散は以下のようになる．

$$
\begin{aligned}
\mathbb{E}(X|y) &= \sum_x x\cdot g(x|y) = \mu_{X|y} \\
\mathbb{E}(Y|x) &= \sum_y y\cdot h(y|x) = \mu_{Y|x} \\
V(X|y) &= \sum_x (x - \mu_{X|y})^2 g(x|y) \\
V(Y|x) &= \sum_y (y - \mu_{Y|x})^2 h(y|x) \\
\end{aligned}
$$

上式は離散型の場合だが，連続型の場合は$\sum \rightarrow \int$にすればよい．

## 独立な確率変数

同時確率分布において，あらゆる$x, y$について

$$
f(x, y) = g(x)h(y)
$$

が成り立つなら，$X, Y$は互いに独立であるという．独立のとき，$g(x)$と$h(y)$について知っていればよく，$f(x, y)$を知る必要はない．独立なので，

$$
g(x|y) \equiv g(x), \quad h(y|x) \equiv h(y)
$$

である．

独立でない場合は，

$$
\begin{aligned}
f(x, y) &= g(x)h(y|x)\\
 &= h(y)g(x|y)
\end{aligned}
$$

が成り立つ．

独立の場合，$x, y$の間に関連はないが，無相関と比べると独立の方がずっと強い．無相関は平均的な性質で，確率分布から計算式により決まる量$\rho$によるが，独立は確率分布そのものに関する仮定である．相関係数はデータから決まるのに対し，独立はそのもの関連なんてないという仮定を置くのだから独立の方が強いでしょう．

独立の場合，積の期待値は

$$
\mathbb{E}(XY) = \mathbb{E}(X)\mathbb{E}(Y)
$$

が成立する．

また，独立なら無相関になるので，

$$
Cov(X, Y) = \mathbb{E}(XY) - \mathbb{E}(X)\mathbb{E}(Y) = 0
$$

となる．逆はそうとは限らない．

さらに，独立な確率変数の和のモーメント母関数は

$$
M_{X+Y}(t) = M_X(t)M_Y(t)
$$

となり，

$$
\mathbb{E}\left(e^{t(x+y)}\right) = \mathbb{E}\left(e^{tX}e^{tY}\right) = \mathbb{E}\left(e^{tX}\right)\mathbb{E}\left(e^{tY}\right)
$$

となる．これはいろいろな定理の証明で中心的な道具として用いられる．

## 多変量正規分布

関連しあう多数の確率変数を最初から仮定して用いられる確率分布のことを多$(n)$次元正規分布と呼ぶ．

導出は，

1. $X_1, X_2$は互いに独立で同じ$\mathcal{N}(0, 1)$に従うとする
1. $Y_1 = aX_1 + bX_2, Y_2 = cX_1 + dX_2$を考える
1. 頑張って同時確率密度関数を求める

の流れになる．最終的に

$$
g(y_1, y_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}\exp\left\{-\frac{1}{2(1-\rho^2)}\left(\frac{y_1^2}{\sigma_1^2} - \frac{2\rho y_1y_2}{\sigma_1\sigma_2} + \frac{y_2^2}{\sigma_2^2}\right)\right\}
$$

が得られる．

## 独立確率変数の和

観測値や測定値などの集計で和をとることはよくある．でも確率変数の和についての確率分布は求めにくい．そこで，独立を仮定するとやや扱いやすくなる．

互いに独立な確率変数の和についての性質をいろいろみていく．

期待値については独立であろうとなかろうと常に加法性

$$
\mathbb{E}(X + Y) = \mathbb{E}(X) + \mathbb{E}(Y)
$$

が成り立つ．

分散については，独立のとき

$$
V(X \pm Y) = V(X) + V(Y)
$$

で（符号に注意！），独立でないときは，

$$
\begin{aligned}
V(X + Y) &= V(X) + V(Y) + 2Cov(X, Y)\\
&= V(X) + V(Y) + 2\rho_{XY}D(X)D(Y)
\end{aligned}
$$

となる．無相関のときは独立のときの式と等価になる．

$X_1, X_2, \dots, X_n$が独立同分布であるなら，$\mathbb{E}(X_i) = \mu, V(X_i) = \sigma^2$とすれば，

$$
\begin{aligned}
\mathbb{E}(X_1+X_2+\cdots X_n) &= n\mu \\
V(X_1+X_2+\cdots X_n) &= n\sigma^2 \\
D(X_1+X_2+\cdots X_n) &= \sqrt{n}\sigma \\
\end{aligned}
$$

となる．

確率変数の相加平均

$$
\bar{X} = \frac{X_1+X_2+\cdots X_n}{n}
$$

の期待値と分散は

$$
\mathbb{E}(\bar{X}) = \mu, \quad V(\bar{X}) = \frac{\sigma^2}{n}
$$

となり，期待値は$n$によらず，分散は$n$が大きくなるほどばらつきがなくなり，安定することがわかる．これは大数の法則に繋がる．
