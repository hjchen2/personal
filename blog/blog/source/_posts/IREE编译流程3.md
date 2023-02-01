---
title: IREE编译流程解析(三)

date: 2023-1-4 20:20:12

category: DL Compiler

tags: [IREE, Deep Learning Compiler]

---

IREE CommonInputConversionPassPipeline主要作用是将IREE::Input dialect lower成IREE::Util、IREE::Flow和IREE::HAL dialect，包括以下几个passes。

<!-- more -->

- createIREEImportPublicPass

将IREE::Input dialect转换成IREE::Util、IREE::Flow和IREE::HAL dialect，并转换func的属性和signature中输入输出类型。比如，

```c++
  iree_input.global private mutable @param  : tensor<1x2xf32>
  func.func @run(%arg0: tensor<1x2xf32>) {
    %0 = iree_input.global.load @param : tensor<1x2xf32>
    %1 = iree_input.tensor.clone %0 : tensor<1x2xf32>
    iree_input.global.store %1, @param : tensor<1x2xf32>
    return
  }
```

转换成（`iree_input.global.load`->`util.global.load`，`iree_input.global.store`->`util.global.store`，`iree_input.tensor.clone`->`flow.tensor.clone`）：

```c++
  util.global private mutable @param : tensor<1x2xf32>
  func.func @run(%arg0: tensor<1x2xf32>) {
    %param = util.global.load @param : tensor<1x2xf32>
    %0 = flow.tensor.clone %param : tensor<1x2xf32>
    util.global.store %0, @param : tensor<1x2xf32>
    return
  }
```

- createImportMLProgramPass

将ml_program dialect转换到IREE::Util dialect。

- createSanitizeModuleNamesPass

  将module name中的`.`替换为`_`，以符合mlir identifiers的命名规范。

  ```c++
  module @iree.module {
    func.func @test(%arg0: f32, %arg1: f32) -> f32 {
      %0 = arith.addf %arg0, %arg1 : f32
      return %0 : f32
    }
  }
  ```

  转换成

  ```c++
  module @iree_module {
    func.func @test(%arg0: f32, %arg1: f32) -> f32 {
      %0 = arith.addf %arg0, %arg1 : f32
      return %0 : f32
    }
  }
  ```
