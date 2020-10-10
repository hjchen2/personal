## ARM NEON指令集

- ARM指令执行Pipeline

<img src="https://github.com/hjchen2/personal/blob/master/images/%E6%88%AA%E5%B1%8F2020-10-1014.35.03.png?raw=true">

- 基本知识

  - NEON寄存器aarch32和aarch64的区别

    - 参考：https://blog.csdn.net/SoaringLee_fighting/article/details/82800919

    - NEON™ Programmer's Guide: https://developer.arm.com/documentation/den0018/a/
    - NEON指令集汇总：https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics

    - Aarch32

      aarch32的neon寄存器组成：

      - 32个S寄存器，S0~S31（单字，32bit）

      - 32个D寄存器，D0~D31（双字，64bit）
      - 16个Q寄存器，Q0~Q15（四字，128bit）

      三种寄存器在物理空间上是复用的，其中

      - S<2n>对应的是D<n>的低半部分，S<2n+1>对应的是D<n>的高半部分
      - D<2n>对应的是Q<n>的低半部分，D<2n+1>对应的是Q<n>的高半部分

      <img src="https://github.com/hjchen2/personal/blob/master/images/%E6%88%AA%E5%B1%8F2020-10-1014.48.40.png?raw=true" alt="image-20201010143232134" style="zoom:50%;" />

       在使用时，如果用到d8~d15（或q4~q7）寄存器，需要先入栈保存vpush {d8-d15}，使用完之后要出栈vpop {d8-d15}。

    - Aarch64

      aarch32的neon寄存器组成：

      - 32个B寄存器，B0~B31，byte，8bit
      - 32个H寄存器，H0~H31，半字，16bit
      - 32个S寄存器，S0~S31，单字，32bit
      - 32个D寄存器，D0~D31，双字，64bit
      - 32个V寄存器，V0~V31，四字，128bit

      同样这五种寄存器在物理空间上也是复用的，其中

      - B<n>对应H<n>的低半部分
      - H<n>对应S<n>的低半部分
      - S<n>对应D<n>的低半部分
      - D<n>对应V<n>的低半部分。

      v0~v7: 用于参数传递和返回值，子程序不需要保存，v8~v15: 子程序调用时必须入栈保存（低64位），v16~v31: 子程序使用时不需要保存。

    - Neon指令可以实现16x8-bit, 8x16-bit, 4x32-bit, 2x64-bit的整型操作，以及8x16-bit*, 4x32-bit, 2x64-bit的浮点操作。

  - 指令集用法差异

    - 32位neon指令都是以V开头，而64位neon指令没有V
    - 32位寄存器需要保存的是r4~r11，q4~q7，而64位寄存器需要保存的是x19~x29 , v8~v15（低64位）
    - 64位下NEON寄存器与32位下NEON寄存器之间的关系不同，32位下d寄存器和q寄存器是重叠映射的，而64下的d寄存器和v寄存器是一一对应的
    - 向64位或者更低位的矢量寄存器中写数据时，会将高位置零
    - 在AArch64中，没有通用寄存器的SIMD和饱和算法指令。只有neon寄存器才有SIMD或饱和算法指令
    - ARM64下没有swap指令和条件执行指令

  - 寄存器命名

    ARMv8 Instruction Set Overview：https://www.element14.com/community/servlet/JiveServlet/previewBody/41836-102-1-229511/ARM.Reference_Manual.pdf

    - Aarch32的寄存器命名

      以小写的寄存器类型加对应的编号组成，比如s0，d1，q2。

      取向量的元素，直接用[]操作符，比如q0[0]表示D0，q0[1]表示D1，q1[0]表示D2。

    - Aarch64的寄存器命名

      以小写的v加对应的V寄存器编号开头，以lanes加位宽为后缀。比如v1.8h表示整个v1寄存器的8个16bit，v2.8B表示v2寄存器的低8个8bit。当bits x lanes小于128时，读取数据时高64bit会被忽略，写入数据时高64bit会被置0。

      <img src="https://github.com/hjchen2/personal/blob/master/images/%E6%88%AA%E5%B1%8F2020-10-1015.37.00.png?raw=true">

      取向量中的元素，指定元素位宽并直接用[]操作符，比如v0.s[0]表示v0寄存器中第一个32bit。

      <img src="https://github.com/hjchen2/personal/blob/master/images/%E6%88%AA%E5%B1%8F2020-10-1015.40.28.png?raw=true">

      当然也支持指定lanes，比如v0.2s[1]，v0.4s[1]，只是结果与v0.s[1]是等价的。

- 8x8矩阵转置的实现（包括aarch32和aarch64）

- aarch64 armv82特性（fp16/sdot/udot）

- 向量dot的实现（包括普通c、aarch32/aarch64 neon、armv82 dot）



## Cache优化

- 基本知识

  - memory hierarchy(L1、L2、local、global)
  - Cache line

- 优化（提高data locality）

  - 分块Tiling

  - 数据预取
  - 循环展开和变换



## CUDA编程

- 基本知识



## AllReduce



## XLA

## TensorRT

## Quantization和低精度

## 控制流

## Transformer模型

## Autotuning



## 算法题

### 递归

- 斐波那契数列

  **斐波那契数列**指的是这样一个数列：

  ![img](https://bkimg.cdn.bcebos.com/formula/791a16929ce9804e0900f0f5bf495f7e.svg)

  这个数列从第3项开始，每一项都等于前两项之和。

  - 递归实现

    ```c++
    #include <iostream>
    
    int f(int N) {
      if (N == 0) return 0;
      if (N == 1) return 1;
      return f(N-1) + f(N-2);
    }
    
    int main() {
      std::cout << f(20) << std::endl;
      return 0;
    }
    ```

  - 尾递归实现

    ```c++
    int tail(int v0, int v1, int n, int N) {
      if (n == N + 1) return v1;
      if (n == 0) return tail(0, 0, n + 1, N);
      if (n == 1) return tail(0, 1, n + 1, N);
      return tail(v1, v0 + v1, n + 1, N);
    }
    
    int main() {
      std::cout << tail(0, 0, 0, 20) << std::endl;
      return 0;
    }
    ```

    由于尾递归函数直接返回函数调用结果，使得调用栈可以被复用，减少栈空间的使用，在N较大时两种方式效率差异非常明显。

- 给一张大面值钞票要兑换成硬币，求有多少种兑换方式

  ```c++
  #include <iostream>
  
  // 硬币种类假设有3种，面值1、2、5。
  static const int coins[3] = {1, 2, 5};
  int count(int value, const int* rest_coins, int rest_coins_count) {
    // 当剩余面值无法被兑换成整数个硬币时，value < 0，此时兑换失败，返回0。
    // 当剩余硬币种类为0时，无法兑换，返回0。
    if (value < 0 || rest_coins_count == 0) return 0;
    
    // 如果value值为0，说明刚好兑换完成，兑换成功，返回1.
    if (value == 0) return 1;
    
    // 当前面值有两种兑换方式：
    //   - 兑换当前硬币，下一次仍然可以兑换当前硬币
    //   - 不兑换当前硬币，之后不再兑换当前硬币
    return count(value，rest_coins + 1，rest_coins_count - 1) + count(value - rest_coins[0], rest_coins, rest_coins_count);
  }
  
  int main() {
    std::cout << count(100) << std::endl;
    return 0;
  }
  ```
