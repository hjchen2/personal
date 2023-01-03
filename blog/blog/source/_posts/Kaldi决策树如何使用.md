---

title: 决策树在Kaldi中如何使用

date: 2016-06-08 14:54:04

category: kaldi, decision tree, 决策树

tags: [kaldi, decision tree, 决策树, HMM, 上下文相关音素]

---

说明：本文是kaldi主页相关内容的翻译（http://kaldi-asr.org/doc/tree_externals.html ）。目前网上已经有一个翻译的版本，但翻译的不是很清楚，导致我在刚学这部分内容的时候产生了一些误解，所以我希望结合我目前所知道的一些东西，尽量把这部分内容翻译地比较容易理解，但由于也是初学者，一些错误也是不可避免，希望大家发现后一起交流，以便我后期修正。好了，还是废话少说吧。

<!-- more -->

## 介绍（Introduction）
本页将对声学决策树在kaldi中如何被创建和使用，以及如何在训练和解码图构建过程进行运用给出一个概述性的解释。对于构建决策树代码的内部描述，请参见Decision tree internals；对于构建解码图方法的详细信息，可以参见Decoding graph construction in Kaldi。

 实现的基本算法就是自顶向下的贪婪分裂，通过问一些问题，比如说左边的音素，右边的音素，中心音素以及当前的状态等等，我们会得到很多可以把数据进行分裂的路径。我们实现的算法与标准算法非常相似，请参见Young，Odell和Woodland的这篇论文"Tree-based State Tying for High Accuracy Acoustic Modeling" 。假设我们对数据建模时采用单高斯将它们分成两部分，在这个算法中，我们通过选择局部最优的问题进行数据分裂，也就是使得似然值增加最大的那个问题。与标准算法实现不同的地方包括可以自由配置树的根节点；对HMM状态和中心音素相关问题提问的能力；以及实际上在Kaldi脚本中默认情况下，问题集是通过对数据自顶向下的二分聚类自动生成的，这就意味着不需要手动去创建问题集。关于树的根节点的配置：可能是把一个共享的群组里面所有音素分裂的统计量，或者独立的音素，或者每个音素的HMM状态，作为树的根节点来进行分裂，或者把音素组作为树的根节点（注：多个音素作为一棵树的根节点）。对于如何用标准的脚本配置根节点，请参见Data preparation。实际上，我们一般让每棵树的根节点都对应一个真实的音素（real phone），意思就是说我们把每个音素的词位置相关、发音相关或者音调相关的所有变种都放进一个音素组，作为决策树的根节点。

本页下面主要给出相关代码层面的一些详细信息。

## 音素上下文窗（Phonetic context windows）
这里我们解释一下在代码中我们怎样描述一个音素的上下文。一棵特殊的决策树将有两个整型值，分别描述的是上下文窗的宽度和中心位置。下表简单说明了这两个值：
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwNjA4MTQwNjU3MTkx?x-oss-process=image/format,png)
N是上下文窗的宽度，P是设计的中心音素的标记。一般P就是窗的中心（因此叫中心位置）；举例说，当N=3时我们一般设P=1，但是我们也可以从0到N-1自由选择；比如，P=2和N=3意味着有左上下文有两个音素，并且没有右上下文。在代码中，当我们讨论中心音素时，我们总是认为讨论的是第P个音素，可能是也可能不是上下文窗中心的那个音素。

一个用来表示典型的triphone上下文窗的整型向量可能是：
```c++
//probably not valid C++
vector<int32> ctx_window = { 12, 15, 21 };
```
假设N=3和P=1，这个表示有一个右上下文21和一个左上下文12的音素15。我们处理端点位置上下文的一个方式是使用0（0不是一个合法的音素，因为在OpenFst中0是为空符号epsilon而保留的），所以比如：
```c++
vector<int32> ctx_window = { 12, 15, 0};
```
表示有一个左上下文12和没有右上下文的音素15，因为音素15是句子的结尾。在句子结尾这种特殊的地方，0这种方式的使用可能有一点意外，因为最后一个“音素”实际上是后续符号“$”（参见Making the context transducer），但是为了在决策树代码中的便利，我们不把后续符号放进上下文窗，而是把0放进去。注意，如果此时我们N=3和P=2，那上述的上下文窗是非法的，因为第P个元素是一个不能表示任何真实音素的0；当然同样如果我们有一个N=1的树，上面的窗都是不合法的，因为那些窗的大小都是错误的。在单音素的情况下，我们可以有一个如下的窗：
```c++
vector<int32> ctx_window = { 15 };
```
因此单音素系统只是被当成上下文相关系统的一种特殊情况，窗的大小N等于1，并且还有一棵什么都不做的树（注：经过这棵树后没有任何参数被绑定）。

## 树的构建过程（The tree building process）
在这部分我们给出Kaldi中树构建过程的一个概述。

即使是单音素系统也有一个决策树，但是是一个无用的树。参见返回这样一个无用树的函数MonophoneContextDependency() 和 MonophoneContextDependencyShared()。这两个函数被命令行程序gmm-init-mono调用；它主要的输入参数是HmmTopology对象，并且输出一棵树，这棵树通常会被以ContextDependency类型的对象写到一个叫做“tree”的文件中，以及模型文件（模型文件包含一个TransitionModel对象和一个AmDiagGmm对象）。如果程序gmm-init-mono接受一个叫-shared-phones的可选参数，它将会在指定的音素序列间共享pdfs（注：输出概率密度函数，比如高斯），否则它会使得所有的音素都是独立的。

从一个扁平的初始（注：除了sil，所有的单音素模型都是一样的）开始训练一个单音素系统后，我们拿单音素对齐的结果和使用函数AccumulateTreeStats()（被acc-tree-stats调用）来累积训练决策树的统计量。这个程序不限于读取单音素的对齐结果；它也能读取上下文相关的对齐结果，因此我们也可以基于triphone对齐结果来构建树。构建树的统计量以BuildTreeStatsType类型（参见Statistics for building the tree）被写到磁盘。函数AccumulateTreeStats()输入N和P的值，N和P就是上文解释过的上下文窗的大小和中心音素位置。命令行程序会默认地将N和P设为3和1，但是也可以使用–context-width和–central-position可选参数进行覆盖。程序acc-tree-stats输入一个上下文无关的音素列表（比如，silence），但是即使存在上下文无关的音素，这个也不是必需的；它只是减少统计量大小的一个机制。对于上下文无关的音素，程序将会累积一个没有定义keys的相关的统计量，keys是跟左右音素对应的（注：在代码中会把一个音素不同的上下文和pdf-class分别作为不同的key，然后累积每个key的统计量）（c.f. Event maps）。

当统计量被积累后，我们使用程序build-tree来构建树。这个程序输出一棵树。程序build-tree需要三样东西：

 - 统计量（BuildTreeStatsType类型）
 - 问题集配置（Questions类型）
 - roots文件（参见下面）

统计量一般从程序acc-tree-stats得到；问题集配置类可以用程序compile-questions输出，compile-questions输入一个声学问题集的拓扑列表（在我们的脚本中，这些都是自动地从构建树的统计量通过程序cluster-phones得到）（注：cluster-phones输入构建树的统计量可以得到一个声学问题集）。roots文件指定了将要在决策树聚类过程中共享根节点的音素集，并且对每个音素集指出下面两个东西：

 - “shared”或者“not-shared”指出是每个pdf-class（也就是一般情况下的HMM状态）都有不同的根节点，还是所有pdf-class共享一个根节点。如果是“shared”，对于所有的HMM状态（比如在正常的HMM拓扑下所有的三个状态）将只会有一个树根节点；如果是“not-shared”，将会有三个树根节点，每个pdf-class有一个。
 - “split”或者“not-split”指出对于根节点要不要根据问题进行决策树分裂（对于silence，我们一般不分裂）。如果该行指定“split”（正常情况），那么我们进行决策树分裂。如果指定“not-split”，那么就不会进行分裂，因此根节点就被无分裂地保留。
 
下面将对这个怎样使用方面做一些阐述：

 - 如果我们指定“shared split”，即使所有的三个HMM状态有一个根节点，不同的HMM状态仍然可以到达不同的叶子节点，因为树可以像对声学上下文的问题提问一样对pdf-class的问题提问。
 - 对于roots文件中同一行出现的所有音素，我们总是让它们共享根节点。如果你不想共享音素的根节点，你只要把它们放在不同的行。
 
下面是roots文件的一个例子；假设音素1是silence，并且其他的音素都有不同的根节点。

```
not-shared not-split  1
shared split  2
...
shared split  28
```
当我们有比如位置和声调相关的音素时，将多个音素放在同一行会非常有用；这样每个“真实的“音素将关联到一个整数的音素ID集合。在这种情况下我们将particular underlying（注：这个不知道怎么翻译）音素的所有变种版本共享一个根节点。下面是来自egs/wsj/s5脚本中Wall Street Journal的roots文件的一个例子（这个例子中音素是用文本表示的，而不是整数形式；但在被Kaldi读取之前会被转换成整数形式（注：就是会把音素映射成整数的ID））：
```
not-shared not-split SIL SIL_B SIL_E SIL_I SIL_S SPN SPN_B SPN_E SPN_I SPN_S NSN NSN_B NSN_E NSN_I NSN_S
shared split AA_B AA_E AA_I AA_S AA0_B AA0_E AA0_I AA0_S AA1_B AA1_E AA1_I AA1_S AA2_B AA2_E AA2_I AA2_S
shared split AE_B AE_E AE_I AE_S AE0_B AE0_E AE0_I AE0_S AE1_B AE1_E AE1_I AE1_S AE2_B AE2_E AE2_I AE2_S
shared split AH_B AH_E AH_I AH_S AH0_B AH0_E AH0_I AH0_S AH1_B AH1_E AH1_I AH1_S AH2_B AH2_E AH2_I AH2_S
shared split AO_B AO_E AO_I AO_S AO0_B AO0_E AO0_I AO0_S AO1_B AO1_E AO1_I AO1_S AO2_B AO2_E AO2_I AO2_S
shared split AW_B AW_E AW_I AW_S AW0_B AW0_E AW0_I AW0_S AW1_B AW1_E AW1_I AW1_S AW2_B AW2_E AW2_I AW2_S
shared split AY_B AY_E AY_I AY_S AY0_B AY0_E AY0_I AY0_S AY1_B AY1_E AY1_I AY1_S AY2_B AY2_E AY2_I AY2_S
shared split B_B B_E B_I B_S
shared split CH_B CH_E CH_I CH_S
shared split D_B D_E D_I D_S
```

当创建这个roots文件时，你应该确保在每一行至少有一个音素是可见的（注：有对应的训练样本）。比如上面的情况，如果音素AY至少在声调和词位置的某些连接中可见，那就没问题。

在这个例子中，对于slience等音素我们有很多的词位置相关的变种。它们将共享它们的pdf's，因为它们都在同一行，并且是“not-split”，但是它们可能会有不同的状态转移参数。实际上，silence的大多数变种都不可能用到，因为silence不可能出现在词与词之间；这只是为了防止以后有人做一些奇怪的事而不会过时。

我们用从之前创建的模型（比如，单音素模型）得到的对齐结果来对混合高斯参数进行初始化；对齐的结果会被程序convert-ali从一棵树转换到另一棵（注：应该就是说对齐的transition不变，但状态绑定的参数可能因为决策树的不同而变化）。

## PDF标号（PDF identifiers）
PDF标号（pdf-id）是一个从0开始的数字，用做概率密度函数（p.d.f.）的序号。系统中每一个p.d.f.都有自己的pdf-id，并且是连续的（在一个LVCSR系统中一般会有几千个）。在树首先被构建时，它们就会被赋值。对于每一个pdf-id对应的是哪个音素，可能知道也可能不知道，这取决于树是怎样被构建的。

## 上下文相关对象（Context dependency objects）
ContextDependencyInterface对象是树的一个虚基类，指定了如何与构建解码图代码进行交互。这个接口只包含四个函数：

 - ContextWidth()返回树需要的N（上下文窗的大小）的值。
 - CentralPosition()返回树需要的P（窗中心位置）的值
 - NumPdfs()返回树定义的pdfs的数量；pdfs的编号从0到NumPdfs()-1。
 - Compute()是对某个特殊的上下文计算它对应的pdf-id的函数
 
  ContextDependencyInterface::Compute()函数的声明如下：
  ```c++
  class ContextDependencyInterface {
   ...
      virtual bool Compute(const std::vector<int32> &phoneseq, int32 pdf_class,
                      int32 *pdf_id) const;
  }
  ```
  如果能计算得到上下文和pdf-class对应的pdf-id，函数返回true。返回false时表明出现了一些错误或者是不匹配。这个函数使用的一个例子：
  ```c++
  ContextDependencyInterface *ctx_dep = ... ;
  vector<int32> ctx_window = { 12, 15, 21 }; // not valid C++
  int32 pdf_class = 1; // probably central state of 3-state HMM.
  int32 pdf_id;
  if(!ctx_dep->Compute(ctx_window, pdf_class, &pdf_id))
        KALDI_ERR << "Something went wrong!"
  else
        KALDI_LOG << "Got pdf-id, it is " << pdf_id;
  ```
  目前唯一继承ContextDependencyInterface的类就是ContextDependency，ContextDependency有少量更丰富的接口；唯一主要的添加就是函数GetPdfInfo，被用于TransitionModel类算出一个特殊的pdf可能对应哪些音素（这个函数的功能可以被 ContextDependencyInterface接口遍历所有的上下文而实现）。

ContextDependency对象实际上是对EventMap对象的简单组合封装；请参见Decision tree internals。我们希望尽可能地隐藏树的真正实现，使得以后需要重构代码时变得非常简单。

## 决策树的一个例子（An example of a decision tree）
决策树文件的格式不是以人们的可读性为首要目标而创建的，但由于大家需要我们在这里试着解释如何去解读这个文件。请看下面的例子，这个是一个来自Wall Street Journal脚本中triphone的决策树。它以这个对象的名字ContextDependency开始（注：在代码中整个树是一个ContextDependency对象）；然后是N（上下文窗的大小），这里是3；接着是P（上下文窗的中心位置），这里是1。文件剩下的部分包含单个EventMap对象。EventMap是一个可能包含指向其他EventMap指针的多态类型。更多详细信息，请参见Event maps。这个文件表示一棵决策树或多棵决策树的集合，并将一个键值对集合（比如，left-phone=5, central-phone=10, right-phone=11, pdf-class=2（注：注意这里是四个键值对，表示一个中心音素是10，上文是音素5，下文是音素11的triphone的第2个状态））映射到一个pdf-id（比如，158）。简单来说，一个决策树包含三种基本类型：一个是SplitEventMap（就像决策树中的分支判断），一个是ConstantEventMap（就像决策树的叶子节点，只包含一个表示pdf-id的数字），和一个是TableEventMap（就像是一个包含其他EventMaps的一个查找表）。SplitEventMap和TableEventMap都有一个需要它们判断的key，这个值可能是0，1或者2，分别表示左上下文音素，中心音素和右上下文音素，也可能是-1，表示pdf-class的标号（注：如果HMM的每个状态都有对应的pdf，则pdf-class可理解为HMM的第几个状态）。一般情况，pdf-class的值与HMM状态的序号是相同的，比如0，1或2。请尝试不要因此而感到困惑：key是-1，value是0，1或2，但它们与上下文窗中音素的keys 0，1或2是没有任何关系的（注：上下文窗中0，1和2表示的是窗中音素的位置）。SplitEventMap有一系列值可以触发决策树的yes分支。下面是一种quasi-BNF符号表示的决策树文件格式。
```
         EventMap := ConstantEventMap | SplitEventMap | TableEventMap | "NULL"
 ConstantEventMap := "CE" <numeric pdf-id>
    SplitEventMap := "SE" <key-to-split-on> "[" yes-value-list "]" "{" EventMap EventMap "}"
    TableEventMap := "TE" <key-to-split-on> <table-size> "(" EventMapList ")"
```
在下面的例子中，树顶层的EventMap是一个以key 1进行分裂的SplitEventMap，也就是按中心音素分裂。在方括号中是一系列连续范围的phone-ids。然而，这些并不表示一个问题，它们只是音素分裂的一种方法，因此我们可以得到每个音素真正的决策树（注：音素真正的决策树是根据音素上下文和pdf-class进行决策的，对中心音素的决策只是为了找到这个音素对应的真正的决策树）。问题在于这棵树是通过“shared roots”方式创建的，所以有很多与同一音素不同词位置和音调标识相关的phone-ids，它们都共享树的根节点。在这种情况下在树的顶层我们不能使用TableEventMap，否则我们就不得不将每棵树重复好几遍（因为EventMap是一棵纯树，而不是一个通用的图，它没有指针共享的机制）。文件后面的一些“SE”标签也是quasi-tree的一部分，它们都是首先按中心音素进行分裂（当我们顺着文件往下看时我们进入了树的更深处；注意这个花括号“{”一直是打开的，还没有关闭）。然后我们看到字符串“TE -1 5 ( CE 0 CE 1 CE 2 CE 3 CE 4 ) ”，表示通过TableEventMap对pdf-class -1进行分裂（实际上就是，HMM-position），并且返回从0到4的值。这5个值表示的是静音和噪声音素SIL，NSN和SPN的5个pdf-ids。在我们的设定中，这三个非语音音素的pdfs是共享的（只有转移矩阵是不同的）。注意：对于这些音素我们用5状态而不是3状态的HMM，所以这里有5个不同的pdf-ids。接下来是“SE -1 [ 0 ] ”，这可以被认为是这棵树中第一个真正的问题。我们可以从上面的SE问题看出这个问题被应用于中心音素为4到19时候，也就是音素AA的不同版本（注：原文写的是5到19，不过我认为原文有问题，改成了4到19）。这个问题问的是pdf-class（key -1）是不是0（即是不是最左边的HMM-state）。下一个问题是“SE 2 [ 220 221 222 223 ]”，问的是音素右上下文是不是音素“M”不同形式中的一个（这是一个非常有效的问题，因为我们是在最左边的HMM-state）；如果问题的答案是yes，我们继续问“SE 0 [ 104 105 106 107... 286 287 ]”，这是一个关于音素左上下文的问题（注：原文写的是右上下文，但应该是左上下文）；如果答案是yes，则pdf-id就是5（“CE 5”），否则就是696（“CE 696”）。
```
s3# copy-tree --binary=false exp/tri1/tree - 2>/dev/null | head -100
ContextDependency 3 1 ToPdf SE 1 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59\
 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 9\
3 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 1\
20 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 14\
5 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170\
 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 \
196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 ]
{ SE 1 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34\
 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 6\
8 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 10\
1 102 103 104 105 106 107 108 109 110 111 ]
{ SE 1 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34\
 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 ]
{ SE 1 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ]
{ SE 1 [ 1 2 3 ]
{ TE -1 5 ( CE 0 CE 1 CE 2 CE 3 CE 4 )
SE -1 [ 0 ]
{ SE 2 [ 220 221 222 223 ]
{ SE 0 [ 104 105 106 107 112 113 114 115 172 173 174 175 208 209 210 211 212 213 214 215 264 265 266 \
267 280 281 282 283 284 285 286 287 ]
{ CE 5 CE 696 }
SE 2 [ 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 132 \
133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 248 249 250 251 252 253 254 255 256 257 2\
58 259 260 261 262 263 268 269 270 271 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 30\
3 ]
```
下面是一个更简单的例子：来自Resource Management脚本的单音素决策树。顶层的EventMap是一个TableEventMap（“TE 0 49 ...”）。key 0是音素位置0，表示中心（并且只有这一个）音素，因为上下文窗大小（N）为1。TE的条目数量是49（音素的数量加1）。表中第一个EventMap是NULL，因为没有序号为0的音素。下一个EventMap是一个有三个元素的TableEventMap，关联到第一个音素的三个HMM状态（技术上来说，是pdf-class）：“TE -1 3 ( CE 0 CE 1 CE 2 )”。
```
s3# copy-tree --binary=false exp/mono/tree - 2>/dev/null| head -5
ContextDependency 1 0 ToPdf TE 0 49 ( NULL TE -1 3 ( CE 0 CE 1 CE 2 )
TE -1 3 ( CE 3 CE 4 CE 5 )
TE -1 3 ( CE 6 CE 7 CE 8 )
TE -1 3 ( CE 9 CE 10 CE 11 )
TE -1 3 ( CE 12 CE 13 CE 14 )
```
## 输入符号信息对象（The ilabel_info object）
CLG图（请参见Decoding graph construction in Kaldi）在它的输入符号位置上有表示上下文相关音素的符号（辅助符号和可能的空符号也一样）。在图中它们总是用整型的标签来表示。在代码和文件名中，我们使用一个叫做ilable_info的对象。ilable_info对象跟ContextFst对象有很密切的联系，请参见see The ContextFst object。就跟许多其他的Kaldi类型一样，ilabel_info也是一个通用的（STL）类型，但是为了可以辨别出是ilabel_info，我们使用与之相同的变量名。就是下面定义的类型：
```c++
std::vector<std::vector<int32> > ilabel_info;
```
它是一个以FST输入标签为索引的vector，给每一个输入标签一个对应的音素上下文窗（参见上文，Phonetic context windows）。比如，假设符号1500是左上下文是12和右上下文是4的音素30，我们有：
```c++
// not valid C++
ilabel_info[1500] == { 4, 30, 12 };
```
在单音素的情况下，我们就会像这样：
```c++
ilabel_info[30] == { 28 }
```
处理辅助符号会有点特殊（参见Disambiguation symbols或者上面引用的Springer Handbook文献，该文献解释了这些辅助符号是什么）。如果一条ilabel_info记录对应到一个辅助符号，我们就把辅助符号的符号表序号取负值放进去（注意这跟辅助符号打印形式#0，#1，#2等等里面的数字是不一样的，它是跟这些辅助符号在符号表文件中的顺序相关的数字，这个符号表文件在我们现在的脚本中叫做phones_disambig.txt）。比如，
```c++
ilabel_info[5] == { -42 }
```
意味着在HCLG中符号数5对应到整数id是42的辅助符号。为了编程方便我们对这些id取负号，因此解析ilable_info对象的程序不需要给一个辅助符号的列表就可以在单音素情况下将它们跟真实的音素进行区分。有两个额外特殊情况：
```c++
ilabel_info[0] == { }; // epsilon
ilabel_info[1] == { 0 }; // disambig symbol #-1;
// we use symbol 1, but don't consider this hardwired.
```
第一个是正常的空符号，我们给它一个空的vector作为它的ilabel_info。这个符号一般不会出现在CLG的左边（注：应该是说不会作为CLG的输入符号）。第二个是一个特殊的辅助符号，打印形式叫做“#-1”。在epsilons被用做标准（Springer Handbook）脚本中C转换器输入符号的时候，我们使用辅助符号“#-1”。它可以确保有空音素表示的词的CLG网络可以被确定化。

程序fstmakecontextsyms可以创建一个与ilabel_info对象打印形式对应的符号表；这个主要用于调试和诊断错误。
