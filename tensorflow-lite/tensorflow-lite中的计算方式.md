## tensorflow-lite中的计算方式

tensorflow-lite对一般的op都提供了两种计算方式，reference和optimized。其中reference的实现定义在tensorflow/contrib/lite/kernels/internal/reference/目录中，optimized的实现定义在tensorflow/contrib/lite/kernels/internal/optimized/目录中。

### 2D Conv      

默认使用optimized

- reference   

  裸写for循环的方式，同时支持float和uint8，uint8转int32计算 
- optimized  
     optimized方式又分成两种实现，float用Gemm，uint8用gemmlowbit
     - Gemm：img2col后利用eigen进行矩阵乘运算，只支持float
     - gemmlowbit: img2col后利用gemmlowbit进行矩阵乘定点计算，目前支持uint8

### DepthwiseConv   

默认使用optimized

- reference   
     裸写for循环的方式，同时支持float和uint8，uint8转int32计算
- optimized
     optimized方式使用Neon指令实现了不同size的kernel，支持float和uint8，需要开启宏ARM_NEON

### FullyConnected   

默认使用PIE

- reference   
     裸写for循环的方式，同时支持float和uint8，uint8转int32计算
- optimized
     optimized方式实现有点多，有以下三种：
     - Gemm: 利用eigen进行矩阵乘运算，只支持float
     - Neon: 目前只支持uint8，需要开启宏ARM_NEON，优先于gemmlowbit
     - gemmlowbit: 利用gemmlowbit进行矩阵乘定点计算，目前支持uint8
- PIE   
     只支持float，uint8使用optimized方式计算
