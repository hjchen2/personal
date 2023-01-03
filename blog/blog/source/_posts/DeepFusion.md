---
title: FusionStitching, Deep Fusion and Code Generation for Tensorflow Computations on GPUs

date: 2019-11-27 12:00:04

category: DL Compiler

tags: [XLA, Deep Learning Compiler, FusionStitching]

---

## FusionStitching系统概述

![屏幕快照 2019-11-25 13.56.40](https://github.com/hjchen2/personal/blob/master/blog/DeepFusion/屏幕快照%202019-11-25%2013.56.40.png?raw=true)

输入HloModule，经过以下三个阶段，最终输出LLVM IR。

- Computation Fusion
- Schedule Planning
- Code Generation

论文主要针对XLA Fusion算法进行了改进，提出了实现Block合并策略的Schedule和Shared Memory Planning技术，以及实现对应的IR Emitter。

<!-- more -->

## Computation Fusion

利用Work/Span analysis，将instruction划分到不同的layer，然后Deep Fusion模块在Schedule Consistency Checker的指导下完成跨layer的instruction合并。该过程是迭代进行的，直到完全没有合并机会。

### Work/Span analysis

> Work/Span analysis通常用于并行算法的分析。假设每个基本运算执行时间都是单位时间，则Work表示的是所有基本运算时间总和，Span表示最长依赖路径上的基本运算时间总和。对于一个计算图来说，可以简单认为图中所有的计算节点总执行时间表示Work，而计算图的最大深度的路径上的节点的顺序执行总时间表示Span。

在这里作者用Span来表示每个节点到root节点的深度。

![屏幕快照 2019-11-26 18.28.17](https://github.com/hjchen2/personal/blob/master/blog/DeepFusion/屏幕快照%202019-11-26%2018.28.17.png?raw=true)

经过Work/Span analysis后，HloModule中的Instruction被划分到了不同的layer，相同Span值的Instruction的layer相同，并且同一layer的Instruction没有依赖关系。

### Subgraph Fusion Algorithm

基于Work/Span analysis计算得到的Span值，作者提出了不同于XLA的Fusion算法。

<img src="https://github.com/hjchen2/personal/blob/master/blog/DeepFusion/屏幕快照%202019-11-26%2019.44.59.png?raw=true" width=600/>

SchdConsistent用来判断fusion_root和hlo是否应该合并，其具体的执行逻辑如下：

1. 如果hlo有一个consumer在giveup集合中，为了防止潜在的循环依赖，放弃fusion。
2. 如果hlo的所有consumer都不在fused集合中，则放弃fusion，因为这里只考虑Producer/Consumer的合并，没有消费关系的Instruction合并则会在ElementwiseFusion算法中处理。
3. 最后会判断合并后的Computation是否存在一个可行的optimized shedule。如果不存在，则放弃fusion。

> - 算法简单高效，Work/Span Analysis的作用其实相当于对Instruction做了一遍拓扑排序，通过简单的合并规则确保circle free。
> - 不区分expensive op，可以通过shared memory来缓存中间结果，因此不需要重计算。
> - 由于第一条约束的强制性，导致合并不完全。

## Schedule Planning

### Schedule定义

> Schedule通常指的是将算法指定的计算过程分配给计算资源的方法。这些计算过程可能包括线程、进程以及数据流等。
>
> 常见的一些Schedule配置:
> - Reorder
> 循环顺序重排，比如for x for y -> for y for x
> - Tile
> - Unroll
> - Vectorize
> - Parallel
> - some CUDA-specific
> 比如blocks、threads、shared memory size等。
>
> 对于包含多个计算stage的算法，Schedule通常是由Consumer驱动，并指定何时对Consumer计算Producer（**Specify when the producer is computed with respect to the consumer** ）。

论文中将Instruction大致分成Elementwise、Transpose、Reduce、BatchDot、Reshape和Broadcast这几种，然后基于这些op定义了一套用来表示对数据分块的Shedule配置。通过一个定义好的Shedule配置和数据的shape，我们就可以知道需要切成多少个数据块，映射到GPU硬件上就是多少个线程块（thread blocks）。

![屏幕快照 2019-11-27 11.22.57](https://github.com/hjchen2/personal/blob/master/blog/DeepFusion/屏幕快照%202019-11-27%2011.22.57.png?raw=true)

Shedule定义在输出shape上，包含三个字段：split_dim、sword和sched_type。split_dim表示切割的维度，取值[0, num_dims)。sword表示在split_dim维度上切分多少块，sword要求能被split_dim维度K整除。sched_type表示行切割还是列切割，取值Row或者Column。给定一个Instruction，其Schedule空间即所有合法的三元组（split_dim、sword和sched_type）。

上图表示Reduce Instruction的两种合法Schedule，通过split_dim和reduce dim来区分Row Schedule和Column Schedule。

### Schedule约束和传播

与Instruction的Schedule定义在输出shape上一样，Computation的Schedule也定义在Root Instruction的输出上，因为Root Instruction是整个Computation的输出。   
对于一个Fused Computation，需要满足Shedule相容约束：即对于Root Instruction，给定一个合法的Shedule，该Shedule需要同时被其他Instruction相容。论文中提出后向传播的方法来判断Shedule约束的相容性。   
任意一个Instruction，其合法的Schedule可以根据Instruction类型和output shape来确定。如果给定的Schedule满足约束（是合法的），则把Schedule后向传播到输入shape(s)，也就是输入Instruction的输出shape。否则从Root Instruction传播过来的Schedule在整个Fused Compution上不满足相容性约束。

> 在Subgraph Fusion算法中，两个Instruction能否合并除了需要满足circle free约束外，还需要满足后端CodeGen模块的支持。只有Schedule满足约束，CodeGen才能正确发射代码，否则CodeGen无法处理。

![屏幕快照 2019-11-27 13.53.21](https://github.com/hjchen2/personal/blob/master/blog/DeepFusion/屏幕快照%202019-11-27%2013.53.21.png?raw=true)

Table.1表明了不同Instruction的Schedule后向传播规则。Schedule约束判断结果会反馈到Subgraph Fusion过程，Fusion不应该破坏Schedule相容性约束。

### Schedule Tuning

任意一个Instruction，split_dim=0和sword=1的Row Schedule总是合法的，也就是只有一个数据块，并且只用一个GPU线程块来计算。这样做的问题也很明显，就是无法充分利用GPU硬件资源。每个Instruction可能有多个合法的Schedule，Schedule Tuning用来选择一个合适的Schedule。  
如果Computation中只有一个Root，遍历该Root Instructon所有合法的满足约束的Schedule，在performance library中查找每个kernel的执行时间，并统计总耗时。总耗时最少的Schedule会被选择用来Code Generation。

如果Computation中有多个Roots，则采取一种two-stage的方法加速Schedule的搜索过程。  
第一步：遍历所有的Roots，计算blocks和blocks对应的Schedule两个序列。对所有Roots对应的blocks序列求交集，结果对应的Schedule即合法的候选Schedule。  
第二步：遍历所有的候选Schedule，计算每个Schedule下所有kernel的耗时，选择耗时最少的Schedule。论文中还提到可以忽略部分ops和early stop的搜索策略，加速第二步的搜索过程。

## Code Generation

### Shared Memory Planning

标记出所有可能需要用到Shared Memory的候选ops，当Memory不足时优先满足most critical ops。

- Size Requirement Analysis

  1. 直接分配
     对于非Root Instruction的Reduce和BatchDot，必须将中间结果放在Shared Memory，allowing consumer ops to use seperate parallel loop emitters to generate code。

  2. 按优先级分配
     对于有多个Users的Elementwise op，为了避免重计算，可以选择将结果缓存到Shared Memory。在memory受限的情况下，按照优先级（expensive op > 非expensive op）确定使用Shared Memory。  
     有时对于只有一个User的expensive op也需要用到Shared Memory，比如如果expensive op后面接了一个BatchDot，由于BatchDot本身对数据的复用性比较高，将expensive op的结果缓存到Shared Memory会带来非常好的性能优化。

- Size Shrinking

  Size Shrinking与上面Requirement Analysis的第2点类似。当每个线程Block分到的数据块非常大时，可能存在Shared Memory受限的问题。解决办法就是让一些ops退化为重计算。  
  从inexpensive ops开始，然后expensive ops，之后是带有BtachDot的expensive ops，最后按照靠近Root Instruction的程度选择候选ops，并优先选择靠近输出的ops。

- Space Sharing

  不同ops分配的Shared Memory是可以复用的，论文中作者提出从Root Instruction开始构造一颗支配树，支配节点可以复用被支配节点申请的Shared Memory。

### Code Generation

XLA使用GpuElementalIrEmitter来实现线程合并的Computation。基于XLA的GpuElementalIrEmitter，作者实现了用于Block合并的IrEmitter (论文中称作IrEmitterStitched)。

<img src="https://github.com/hjchen2/personal/blob/master/blog/DeepFusion/屏幕快照%202019-11-27%2017.26.12.png?raw=true" width=600/>

IrEmitterStitched输入有hlo、root、shared、schedule和generators。

- hlo: 待处理的hlo Instruction
- root: 是否是root Instruction
- shared: 是否将输出写到shared memory
- shedule: row schedule还是column schedule
- generators：与XLA GpuElementalIrEmitter中的generators_类似，但是能处理shared memory的情况。

基本逻辑如下：

1. 如果待处理的Instruction不是root Instruction，并且没有用到Shared Memory，不是Dot和Reduce Opcode，则回退到XLA的GpuElementalIrEmitter中去处理，否则使用IrEmitterStitched发射LLVM代码。
2. 如果需要用到Shared Memory，则调用EmitWriteSharedArray将结果写到Shared Memory。
3. 如果是root Instruction，则调用EmitWriteOutputArray将结果写到Global Memory。如果不是root Instruction，则调用EmitGenerator在generators中创建一个entry，以支持当前Instruction与其他Instruction的合并。



## XLA op fusion规则

- Consumer本身支持合并

  特定op不支持与Producer合并，比如Parameter、While、Conditional、Call等，以及op本身has a side effect或者调用了has a side effect的op。此外被标记为tracing的op也无法合并。

- Consumer与Producer之间支持合并

  - Consumer和Producer之间所有的op均可以被合并到Consumer。
  - 对于Consumer和Producer之间所有的op：
    1. 如果直接Producer已经是一个Fusion op，则不能合并。
    2. 对Reduce和Scatter，以及CustomCall/LibraryCall的一些限制。
    3. 如果直接Producer有其他Consumer，则Fusion会导致该Producer 需要重计算。如果Producer属于expensive op或为Parameter op则放弃合并。
