---
title: IREE编译流程解析(二)

date: 2023-1-4 20:14:12

category: DL Compiler

tags: [IREE, Deep Learning Compiler]

---

IREE InputConversionPassPipeline的主要作用是将不同的输入（MHLO、XLA、Torch Tensor和TOSA）统一lower成linalg dialect和builtin的arith dialect、scf dialect和tensor dialect。下面以MHLO输入为例，列举了InputConversionPassPipeline中各个pass以及它们的主要作用。

<!-- more -->

- mhlo::createLegalizeControlFlowPass

  将TF1.0中的控制流原语（http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf ）规范化成HLO中的控制流算子。

- createTopLevelSCFToCFGPass

- createMHLOToMHLOPreprocessingPass

- mlir::createCanonicalizerPass

- mlir::createShapeToShapeLowering

  将 `shape.num_elements` 转换成 `shape.reduce`。

- mlir::createConvertShapeToStandardPass

  将shape dialect lower成arith dialect、scf dialect和tensor dialect。比如

  ```c++
  func.func @test(%arg0: tensor<1x?xf32>, %arg1: tensor<?xf32>) -> index {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = shape.dim %arg0, %c1 : tensor<1x?xf32>, index -> index
    %1 = shape.dim %arg1, %c0 : tensor<?xf32>, index -> index
    %2 = shape.add %0, %1 : index, index -> index
    return %2 : index
  }
  ```

  转换成

  ```c++
  func.func @test(%arg0: tensor<1x?xf32>, %arg1: tensor<?xf32>) -> index {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %0 = tensor.dim %arg0, %c1_1 : tensor<1x?xf32>
      %1 = tensor.from_elements %c1_0, %0 : tensor<2xindex>
      %2 = tensor.cast %1 : tensor<2xindex> to tensor<2xindex>
      %3 = tensor.dim %arg0, %c1 : tensor<1x?xf32>
      %c0_2 = arith.constant 0 : index
      %4 = tensor.dim %arg1, %c0_2 : tensor<?xf32>
      %5 = tensor.from_elements %4 : tensor<1xindex>
      %6 = tensor.cast %5 : tensor<1xindex> to tensor<1xindex>
      %7 = tensor.dim %arg1, %c0 : tensor<?xf32>
      %8 = arith.addi %3, %7 : index
      return %8 : index
    }
  ```

- mlir::createCanonicalizerPass

- mlir::createInlinerPass

  内联calls和callable operations，并删除dead callables。比如：

  ```c++
  func.func @test(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = call @add(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  func.func private @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  ```

  私有的add函数被内联之后删除，

  ```c++
  func.func @test(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  ```

- IREE::Util::createDemoteI64ToI32Pass

- IREE::Util::createDemoteF64ToF32Pass

- mlir::createCanonicalizerPass

- mlir::createCSEPass

- mhlo::createLegalizeShapeComputationsPass

  把scalar tensor op转换成scalar op + fromElements op。比如

  ```c++
  func.func @test(%arg0: f32, %arg1: f32) -> tensor<1xf32> {
    %0 = tensor.from_elements %arg0 : tensor<1xf32>
    %1 = tensor.from_elements %arg1 : tensor<1xf32>
    %2 = mhlo.add %0, %1 : tensor<1xf32>
    return %2 : tensor<1xf32>
  }
  ```

  转换成：

  ```c++
  func.func @test(%arg0: f32, %arg1: f32) -> tensor<1xf32> {
    %0 = arith.addf %arg0, %arg1 : f32
    %1 = tensor.from_elements %0 : tensor<1xf32>
    return %1 : tensor<1xf32>
  }
  ```

- createConvertMHLOToLinalgExtPass

  将`mhlo::sort`、`mhlo.scatter`、`mhlo.fft`、`mhlo.reverse`、`mhlo.topk`转换到IREE::LinalgExt dialect，同时将在IREE::LinalgExt dialect区域内部的mhlo op转换成linalg dialect，`mhlo.return`则转换成`iree_linalg_ext.yield`。比如，

  ```c++
  func.func @test(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "mhlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = mhlo.compare  GT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      mhlo.return %1 : tensor<i1>
    }) {dimension = 0 : i64} : (tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
  ```

  转换成，

  ```c++
  func.func @test(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = iree_linalg_ext.sort dimension(0) outs(%arg0 : tensor<10xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.cmpf ogt, %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : i1
    } -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
  ```

- createMHLOToLinalgOnTensorsPass

  将外层剩余的mhlo op转换到linalg dialect。比如

  ```c++
  func.func @test(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  ```

  转换成，

  ```c++
  func.func @test(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = linalg.init_tensor [1] : tensor<1xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %2 = arith.addf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    } -> tensor<1xf32>
    return %1 : tensor<1xf32>
  }
  ```

- mlir::createReconcileUnrealizedCastsPass

  消除unrealized conversion cast操作。算法过程描述：如果unrealized conversion cast是dead节点（没有user或所有users也都是unrealized conversion cast），则直接删除该dead节点；如果是live节点（至少有一个非unrealized conversion cast的user），则遍历其所有子节点，如果其子节点中所有unrealized conversion cast的result type与该op的input type相同（即不存在真实意义的type cast操作），则将所有遍历到的unrealized conversion cast都折叠成该op的输入，否则报错`live unrealized conversion cast`。

- mlir::createCanonicalizerPass

- createVerifyCompilerMHLOInputLegality

  将mhlo、chlo、shape dialect设为非法dialect，验证lower过程是否完全。
