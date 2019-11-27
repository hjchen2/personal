---

title: NEURAL MACHINE TRANSLATION论文学习串讲

date: 2017-12-01 12:24:08

category: neural machine translation

tags: [seq2seq, machine translation, Encoder-Decoder, Attention]

---

## seq2seq

主要学习的是论文Neural machine translation by jointly learning to align and translate (Dzmitry Bahdanau、Yoshua Bengio等，2016.05)和Neural machine translation (Minh-ThangLuong，2016.12)。

神经机器翻译的目的是将一门语言的文本序列翻译成另一门语言的文本序列，因此机器翻译的训练语料一般是源语言和目标语言组成的一对文本，也叫做平行语料(parallel corpus)。我们通常将输入和输出都是序列的模型叫做seq2seq，seq2seq不仅应用在机器翻译领域，也用于当前热门的自动问答系统以及文本摘要的自动生成等领域。



## Encoder-Decoder

2014年Dzmitry Bahdanau、Yoshua Bengio等人在论文Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation中首次提出将RNN Encoder-Decoder结构来计算双语短语对的条件概率，用于改进统计机器翻译的效果。Encoder-Decoder是由encoder和decoder两部分组成，encoder将输入序列编码成定长的语义向量，decoder将语义向量进行解码得到目标序列。

<img src="https://github.com/hjchen2/personal/blob/master/blog/nmt/12c7a5370bc9da07193c0bd43c5b27cd.png?raw=true" width=500 align=center>

在NMT中Encoder-Decoder试图直接对并行语料的条件概率$P(Y|X)$进行建模，encoder输入的是一组向量序列$X=(x_{1},…,x_{T_{x}})$，$x_i$为词$i$的one-hot编码向量，并将序列$X$编码成语义向量$c$，decoder输入语义向量$c$，并逐个生成序列$Y=(y_{1},…,y_{T_{y}})$，其中$y_{i}$的生成与之前已经生成的词序列$y_{1},…,y_{i-1}$有关。

$$\log p(Y|X)=\sum_{t=1}^{T_{y}}\log p(y_{t}|y_{<t}, c)$$

对于不定长度序列的编码和解码，我们很自然会想到RNN，实际上RNN Encoder–Decoder就是正反两组RNN拼接在一起组成的编码解码网络。经典的RNN Encoder–Decoder示意图如下：

<img src="https://github.com/hjchen2/personal/blob/master/blog/nmt/ab3551f2c0f12a3bc50283e49e09e52c.png?raw=true" width=400 align=center />

我们可以用下面公式描述编码过程：
$$h_{t}=f(x_{t},h_{t-1})$$
$$c=q({h_{1},…,h_{T_{x}}})$$

函数$f$一般用一个RNN结构来表示，可以是LSTM、GRU等，$h_{t}$表示encoder RNN在第t时刻的cell隐状态，向量c的计算与encoder RNN所有时刻的cell隐状态相关，函数$q$可以表示所有隐状态的加权和，但由于RNN的特殊性，我们这里只使用最后一个时刻的隐状态作为向量$c$，即$c=h_{T_{x}}$。

对于解码过程，生成$y_{t}$时的条件概率可以改写成

$$p(y_{t}|y_{<t},c)=g(y_{t-1},s_{t},c)$$
$$s_{t}=f(s_{t-1},y_{t-1},c)$$

其中，$g$是非线性函数，可以是单层的softmax，也可以是一个多层结构的神经网络，$y_{t-1}$表示上一时刻的输出，$f$同样是一个RNN结构，$s_{t}$表示decoder RNN cell的隐状态。



## Attention

在Encoder-Decoder中每个目标词生成时使用的都是同一个向量$c$，虽然理论上来讲向量$c$可以表示输入序列的语义信息，比如一些关键词、句子结构和语法信息等，但也存在注意力分散的问题。在机器翻译中，一般翻译出来的词与源序列的词是有对齐关系的，也就是说目标词的生成与源序列中的部分关键词关系更大，而其他词对当前目标词的生成影响就很小。在Encoder-Decoder中不论生成哪个目标词，使用的语义向量都是$c$，而语义向量$c$是由句子$X$的每个单词经过Encoder编码而成的，也就意味着句子$X$中的关键词对生成任意目标词的影响力是相同的。

<img src="https://coding.net/u/hjchen2/p/personal/git/raw/master/blog/pictures/v2-db380a8bf032afa9533d358389de99d6_hd.jpg?raw=true" width=500>



第一篇论文在Encoder-Decoder的基础上引入注意力机制，来解决上述注意力分散的问题。在论文中提出，每个目标词生成时使用的语义向量是不同的，也就是说Encoder-Decoder将会学会在生成目标词时给每个源语词分配权重，这个权重表示该源语词对当前目标词的重要程度。增加了attention机制的Encoder-Decoder框架如下图：

<img src="https://github.com/hjchen2/personal/blob/master/blog/nmt/e9ba93ee15054825cb2c66a7180ef022.png?raw=true" width=400 align=center>

在基于attention的模型中，每个目标词生成时的条件概率可以写成：
$$p(y_{i}|y_{<t},X)=g(y_{i-1},s_{i},c_{i})$$
$$s_{i}=f(s_{i-1},y_{i-1},c_{i})$$

在RNN中每个时刻的隐状态$h_{i}$可以表示第$i$个源语词及其周围部分词的信息，因此与之前的Encoder-Decoder框架不同，语义向量$c_{i}$不再是encoder RNN最后一个时刻的隐状态，而是与encoder RNN所有时刻的隐状态（$h_{1},...,h_{T_{x}}$）相关的一个向量。

$$c_{i}=\sum_{j=1}^{T_{x}}\alpha_{ij}h_{j}$$
$\alpha_{ij}$可以认为是目标词$i$与源语词$j$的对齐权重，因此可以使用源语词$i$的隐状态$h_{i}$和目标词前一时刻的隐状态$s_{i-1}$来计算。
$$\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T_{x}}\exp(e_{ik})}$$
其中
$$e_{ij}=a(s_{i-1},h_{j})$$
$a$是一个对齐模型，在Bahdanau的论文中将其定义成一个前馈神经网络，与Encoder-Decoder一起参与训练。计算公式如下：
$$a(s_{i-1},h_{j})=v_{a}^\mathsf{T}\cdot tanh(W_{a}s_{i-1}+U_{a}h_{j}) $$
$v_{a}$、$W_{a}$和$U_{a}$都是对齐模型的参数。在第二篇ThangLuong的论文中提出下面三种计算方式，本质上也是大同小异。

<img src="https://github.com/hjchen2/personal/blob/master/blog/nmt/667d0e7417d384138f961490ff0745c3.png?raw=true" width=400 align=center>

下图是Bahdanau在论文中给出的一个模拟图，图中模拟的是在给定源语序列（$X_{1},X_{2},...,X_{T}$）的情况下生成第$t$个目标词$y_{t}$的过程。

<img src="https://github.com/hjchen2/personal/blob/master/blog/nmt/970f70807791925f3f8f54266e0a8435.png?raw=true" width=300 align=center>



## Encoder

在Bahdanau的论文中Encoder和Decoder使用的都是GRU（Gated Recurrent Unit），GRU与LSTM一样都是RNN众多变体中比较常见的一种，也可以使用其他变体RNN，比如在ThangLuong的论文中主要用的就是LSTM。

我们知道传统的RNN理论上可以记忆无限长的序列，但由于递归权重对每个时刻的输入都是一样的，这就导致一个二选一的问题：(1) 模型发散，无法收敛（2）梯度消失，无法产生长时记忆。GRU和LSTM一样，都是通过引入门（gate）的机制来解决传统RNN梯度消失的问题，gate打开和关闭是由当前时刻的输入和前一时刻的隐层状态控制的，也就是说每个时刻gate的状态都是不同的，一些需要长时间记忆的信息会通过gate一直传递下去，从而学习到长距离依赖。

传统RNN的隐层计算公式：$h_{t}=g(W^{hh}h_{t-1}+W^{hx}x_{t})$，$W^{hh}$是递归权重，$W^{hx}$是隐层的权重。实际上，LSTM和GRU都可以认为是对$h_{t}$计算方式的改进。

下面是GRU结构的示意图，输入为$h_{t-1}$和$x_{t}$，输出为$h_{t}$。在GRU中存在两个gate，一个是reset gate，一个是update gate，分别对应下图中的$r_{t}$和$z_{t}$，$\widetilde h_{t}$表示候选隐层状态，候选隐层状态与上一时刻的隐层状态$h_{t-1}$一起更新当前时刻的隐层状态$h_{t}$。

<img src="https://coding.net/u/hjchen2/p/personal/git/raw/master/blog/pictures/rnn-gru-unit.png?raw=true" width=400 align=center>

GRU的计算过程：   
1、首先计算重置门$r_{t}$和更新门$z_{t}$，其中$\sigma$表示sigmoid函数
$$r_{t}=\sigma(W^{r}x_{t}+U^{r}h_{t-1})$$
$$z_{t}=\sigma(W^{z}x_{t}+U^{z}h_{t-1})$$
2、计算候选隐层状态$\widetilde h_{t}$，其中$r_{t}$用来控制历史记忆的传递，如果$r_{t}=0$，那么$\widetilde h_{t}$只与当前输入$x_{t}$有关，历史记忆被重置。
$$\widetilde h_{t}=tanh(Wx_{t}+U[r_{t}\odot h_{t-1}])$$
实际上仅仅增加一个reset gate就已经可以解决长时依赖的问题，因为如果有需要$r_{t}$可以总等于1，那么历史记忆就会一直传递下去。但这会带来一个问题，$h_{t-1}$会累加到当前时刻的隐层状态上产生新的记忆，不断累加的记忆会导致$\widetilde h_{t}$达到饱和，最终导致模型无法收敛。为了解决这个问题，GRU可以选择对当前输入产生的新记忆进行遗忘，只传递之前的历史记忆，也就是说我们允许GRU舍弃一些对后续无关的输入信息，保证记忆都是有效信息。GRU是通过下面的更新操作来实现这个过程的，
$$h_{t}=z_{t}\odot h_{t-1}+(1-z_{t})\odot \widetilde h_{t}$$
$z_{i}$反映了相对历史记忆当前输入信息的重要程度，$z_{i}$越小表明当前输入信息越重要。

实际上在Bahdanau的论文中使用的是双向RNN（BiRNN），BiRNN在前向RNN的基础上增加了一个反向RNN，使得RNN可以同时看到历史和未来的信息，最终前向RNN的隐层状态和反向RNN的隐层状态拼接后输出。

$$h_{i}=\left [ \begin{align} & \vec{h_{i}} \\ & \stackrel{\leftarrow}{h_{i}} \end{align}\right ]$$



## Decoder

在Bahdanau的论文中decoder采用是一个前向的GRU，但与encoder GRU不同的是decoder GRU需要额外输入语义向量$c_{i}$。decoder GRU隐层状态$s_{i}$的计算如下：
$$s_{i}=(1-z_{i})\odot s_{i-1}+z_{i}\odot \widetilde s_{i}$$
其中，   
$$\widetilde s_{i}=tanh(Wy_{i-1}+U[r_{i}\odot s_{i-1}]+Cc_{i})$$
$$r_{i}=\sigma(W_{r}y_{i-1}+U_{r}s_{i-1}+C_{r}c_{i})$$
$$z_{i}=\sigma(W_{z}y_{i-1}+U_{z}s_{i-1}+C_{z}c_{i})$$
encoder GRU的隐层状态会被传递到decoder GRU用于生成第一个目标词，所以decoder GRU的隐层状态的初始值不是0，而是将encoder中反向GRU第一个时刻的隐层状态直接复制给decoder GRU，即$s_{0}=tanh(W_{s}\stackrel{\leftarrow}{h_{1}})$。



## beam search
