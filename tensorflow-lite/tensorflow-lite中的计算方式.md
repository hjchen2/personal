## tensorflow-lite中的计算方式

tensorflow-lite对一般的op都提供了两种计算方式，reference和optimized。其中reference的实现定义在tensorflow/contrib/lite/kernels/internal/reference/目录中，optimized的实现定义在tensorflow/contrib/lite/kernels/internal/optimized/目录中。

### 2D Conv      
- reference: 裸写for循环的方式，同时支持float和uint8，uint8转int32计算
- Gemm：img2col后利用eigen进行矩阵乘运算，只支持float
- gemmlowbit: img2col后利用gemmlowbit进行矩阵乘定点计算，目前支持uint8

### DepthwiseConv   
- reference：裸写for循环的方式，同时支持float和uint8，uint8转int32计算
- Neon: 实现了不同size的kernel，支持float和uint8

### FullyConnected   
- reference: 裸写for循环的方式，同时支持float和uint8，uint8转int32计算
- PIE: 只支持float
- Gemm: 利用eigen进行矩阵乘运算，只支持float
- Neon: 目前只支持uint8
- gemmlowbit: 利用gemmlowbit进行矩阵乘定点计算，目前支持uint8
