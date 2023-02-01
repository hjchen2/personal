---
title: IREE编译流程解析(四)

date: 2023-1-4 20:30:12

category: DL Compiler

tags: [IREE, Deep Learning Compiler]

---

IREE ABI::TransformPassPipeline主要作用是将外部导入的接口和本module导出到外部的接口参数统一成标准标量类型或`hal.buffer_view`类型（`hal.buffer_view`对应tensor），包含以下几个passes。

<!-- more -->

- createWrapEntryPointsPass

  给external func生成一个内部函数，函数中调用原始的external func，同时将public func的函数体包装成一个新的函数，原public func中调用该函数。该pass最终的目的是将外部导入的接口和本module导出到外部的接口参数统一成标准标量类型或`hal.buffer_view`（`hal.buffer_view`对应tensor类型）。

  ```c++
    // external/imported func
    func.func private @add(tensor<f32>, tensor<f32>) -> tensor<f32>
    
    // public/exported func
    func.func @test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = call @add(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      return %0 : tensor<f32>
    }
  ```

  转换成，

  ```c++
    func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
    func.func private @_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = hal.tensor.export %arg0 : tensor<f32> -> !hal.buffer_view
      %1 = hal.tensor.export %arg1 : tensor<f32> -> !hal.buffer_view
      %2 = call @add(%0, %1) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
      %3 = hal.tensor.import %2 : !hal.buffer_view -> tensor<f32>
      return %3 : tensor<f32>
    }
    func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<f32>
      %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<f32>
      %2 = call @_test(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %3 = hal.tensor.export %2 : tensor<f32> -> !hal.buffer_view
      return %3 : !hal.buffer_view
    }
    func.func private @_test(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = call @_add(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      return %0 : tensor<f32>
    }
  ```

- mlir::createInlinerPass

  将WrapEntryPointsPass中生成的wrap函数内联起来。最终转换成，

  ```c++
    func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
    func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %0 = call @add(%arg0, %arg1) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
      return %0 : !hal.buffer_view
    }
  ```

- mlir::createCanonicalizerPass

- mlir::createCSEPass

- mlir::createSymbolDCEPass
