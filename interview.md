## ARM NEON指令集

- 基本知识

  - NEON寄存器

  - 指令命名规则
  - aarch32和aarch64的区别

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
  
  # 硬币种类假设有3种，面值1、2、5。
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
