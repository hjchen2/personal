---
title: IREE编译流程解析(五)

date: 2023-1-4 21:15:20

category: DL Compiler

tags: [IREE, Deep Learning Compiler]

---

IREE Flow::buildFlowTransformPassPipeline主要作用是执行一系列窥孔优化，比如1x1的conv2d转换成matmul、tiling、op fusion等，最终将workload拆分成`flow.executable`。相关的passes及其作用如下。

<!-- more -->

- IREE::Util::createDemoteF64ToF32Pass

  将F64类型窄化为F32。

- IREE::Flow::createConvertConv2D1x1ToMatmulPass

  将1x1的`linalg.conv_2d_nhwc_hwcf`转换成`linalg.matmul`。

  ```c++
    // func.func @conv(%input : tensor<1x2x2x3xf32>, %filter: tensor<1x1x3x4xf32>) -> tensor<1x2x2x4xf32> {
    //   %0 = mhlo.convolution(%input, %filter)
    //             dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    //             window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]}
    //             {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
    //           : (tensor<1x2x2x3xf32>, tensor<1x1x3x4xf32>) -> tensor<1x2x2x4xf32>
    //   return %0 : tensor<1x2x2x4xf32>
    // }
    func.func @conv(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x2x2x3xf32>
      %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<1x1x3x4xf32>
      %2 = linalg.init_tensor [1, 2, 2, 4] : tensor<1x2x2x4xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32>
      %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<1x2x2x3xf32>, tensor<1x1x3x4xf32>) outs(%3 : tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32>
      %5 = hal.tensor.export %4 : tensor<1x2x2x4xf32> -> !hal.buffer_view
      return %5 : !hal.buffer_view
    }
  ```

  转换成，

  ```c++
  func.func @conv(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x2x2x3xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<1x1x3x4xf32>
    %2 = linalg.init_tensor [1, 2, 2, 4] : tensor<1x2x2x4xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32>
    %4 = tensor.collapse_shape %0 [[0, 1, 2], [3]] : tensor<1x2x2x3xf32> into tensor<4x3xf32>
    %5 = tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<1x1x3x4xf32> into tensor<3x4xf32>
    %6 = tensor.collapse_shape %3 [[0, 1, 2], [3]] : tensor<1x2x2x4xf32> into tensor<4x4xf32>
    %7 = linalg.matmul ins(%4, %5 : tensor<4x3xf32>, tensor<3x4xf32>) outs(%6 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %8 = tensor.expand_shape %7 [[0, 1, 2], [3]] : tensor<4x4xf32> into tensor<1x2x2x4xf32>
    %9 = hal.tensor.export %8 : tensor<1x2x2x4xf32> -> !hal.buffer_view
    return %9 : !hal.buffer_view
  }
  ```

- IREE::Flow::createConvertConv2DToImg2ColPass

  将conv2d转换成img2col。默认不开启。

  ```c++
  // %0 = mhlo.convolution(%input, %filter)
  //               dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
  //               window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]}
  //               {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  //             : (tensor<1x4x4x3xf32>, tensor<2x2x3x4xf32>) -> tensor<1x3x3x4xf32>
  func.func @conv(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x4x4x3xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2x2x3x4xf32>
    %2 = linalg.init_tensor [1, 3, 3, 4] : tensor<1x3x3x4xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<1x4x4x3xf32>, tensor<2x2x3x4xf32>) outs(%3 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %5 = hal.tensor.export %4 : tensor<1x3x3x4xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
  ```

  转换成，

  ```c++
  func.func @conv(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x4x4x3xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2x2x3x4xf32>
    %2 = linalg.init_tensor [1, 3, 3, 4] : tensor<1x3x3x4xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %4 = linalg.init_tensor [1, 3, 3, 2, 2, 3] : tensor<1x3x3x2x2x3xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x4x4x3xf32>) outs(%4 : tensor<1x3x3x2x2x3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<1x3x3x2x2x3xf32>
    %6 = tensor.collapse_shape %5 [[0, 1, 2], [3, 4, 5]] : tensor<1x3x3x2x2x3xf32> into tensor<9x12xf32>
    %7 = tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<2x2x3x4xf32> into tensor<12x4xf32>
    %8 = tensor.collapse_shape %3 [[0, 1, 2], [3]] : tensor<1x3x3x4xf32> into tensor<9x4xf32>
    %9 = linalg.matmul ins(%6, %7 : tensor<9x12xf32>, tensor<12x4xf32>) outs(%8 : tensor<9x4xf32>) -> tensor<9x4xf32>
    %10 = tensor.expand_shape %9 [[0, 1, 2], [3]] : tensor<9x4xf32> into tensor<1x3x3x4xf32>
    %11 = hal.tensor.export %10 : tensor<1x3x3x4xf32> -> !hal.buffer_view
    return %11 : !hal.buffer_view
  }
  ```

- IREE::Flow::createDetachElementwiseFromNamedOpsPass

  将`buffer = linalg.generic_op + linalg.named_payload_op`转换成`tmp_buffer = linalg.named_payload_op; buffer =  linalg.generic_op + tmp_buffer`，主要目的是将上游的`generic op`和`named_payload_op`分隔开，使得`named_payload_op`的结果写到一块新的buffer。

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x4x4x3xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2x2x3x4xf32>
    %2 = hal.tensor.import %arg2 : !hal.buffer_view -> tensor<1x3x3x4xf32>
    
    %3 = linalg.init_tensor [1, 3, 3, 4] : tensor<1x3x3x4xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x3x3x4xf32>) outs(%4 : tensor<1x3x3x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %8 = arith.addf %arg3, %arg3 : f32
      linalg.yield %8 : f32
    } -> tensor<1x3x3x4xf32>
    
    %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<1x4x4x3xf32>, tensor<2x2x3x4xf32>) outs(%5 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %7 = hal.tensor.export %6 : tensor<1x3x3x4xf32> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
  ```

  转换成，

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x4x4x3xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2x2x3x4xf32>
    %2 = hal.tensor.import %arg2 : !hal.buffer_view -> tensor<1x3x3x4xf32>
    
    %3 = linalg.init_tensor [1, 3, 3, 4] : tensor<1x3x3x4xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x3x3x4xf32>) outs(%4 : tensor<1x3x3x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %11 = arith.addf %arg3, %arg3 : f32
      linalg.yield %11 : f32
    } -> tensor<1x3x3x4xf32>
    
    %6 = linalg.init_tensor [1, 3, 3, 4] : tensor<1x3x3x4xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<1x4x4x3xf32>, tensor<2x2x3x4xf32>) outs(%7 : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
  
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %5 : tensor<1x3x3x4xf32>, tensor<1x3x3x4xf32>) outs(%7 : tensor<1x3x3x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %11 = arith.addf %arg3, %arg4 : f32
      linalg.yield %11 : f32
    } -> tensor<1x3x3x4xf32>
    %10 = hal.tensor.export %9 : tensor<1x3x3x4xf32> -> !hal.buffer_view
    return %10 : !hal.buffer_view
  }
  ```

- IREE::Flow::createVerifyInputLegalityPass

  检查tosa dialect、mhlo dialect和UnrealizedConversionCastOp是否已经被转化完全。

- IREE::Flow::createConvertLinalgMatmulToMmt4DPass

  将2d的`linalg.matmul` tiling成`linalg.mmt4d`。默认不开启，可通过`--iree-flow-mmt4d-target-options="enable_generic_slow arch=cuda"`选项开启。

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<128x256xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<256x256xf32>
    %2 = linalg.init_tensor [128, 256] : tensor<128x256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<128x256xf32>, tensor<256x256xf32>) outs(%3 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %5 = hal.tensor.export %4 : tensor<128x256xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
  ```

  转换成，

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<128x256xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<256x256xf32>
    %2 = linalg.init_tensor [128, 256] : tensor<128x256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %4 = tensor.expand_shape %0 [[0, 1], [2, 3]] : tensor<128x256xf32> into tensor<16x8x128x2xf32>
    %5 = tensor.expand_shape %1 [[0, 1], [2, 3]] : tensor<256x256xf32> into tensor<128x2x64x4xf32>
    %6 = tensor.expand_shape %3 [[0, 1], [2, 3]] : tensor<128x256xf32> into tensor<16x8x64x4xf32>
    %7 = linalg.init_tensor [16, 128, 8, 2] : tensor<16x128x8x2xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<16x8x128x2xf32>) outs(%7 : tensor<16x128x8x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<16x128x8x2xf32>
    %9 = linalg.init_tensor [64, 128, 4, 2] : tensor<64x128x4x2xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<128x2x64x4xf32>) outs(%9 : tensor<64x128x4x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<64x128x4x2xf32>
    %11 = linalg.init_tensor [16, 64, 8, 4] : tensor<16x64x8x4xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<16x8x64x4xf32>) outs(%11 : tensor<16x64x8x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<16x64x8x4xf32>
    // 16 x (128x8x2) @ 64 x (128x4x2) => 16 x 64 x sum_{128}(8x2 * (4x2)^T)
    %13 = linalg.mmt4d {comment = "generic tiling parameters, as no known kernel was matched for this matmul and target"} ins(%8, %10 : tensor<16x128x8x2xf32>, tensor<64x128x4x2xf32>) outs(%12 : tensor<16x64x8x4xf32>) -> tensor<16x64x8x4xf32>
    %14 = linalg.init_tensor [16, 8, 64, 4] : tensor<16x8x64x4xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<16x64x8x4xf32>) outs(%14 : tensor<16x8x64x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<16x8x64x4xf32>
    %16 = tensor.collapse_shape %15 [[0, 1], [2, 3]] : tensor<16x8x64x4xf32> into tensor<128x256xf32>
    %17 = hal.tensor.export %16 : tensor<128x256xf32> -> !hal.buffer_view
    return %17 : !hal.buffer_view
  }
  ```

- IREE::Flow::createPadLinalgOpsToIntegerMultiplePass

  将matmul的M、N和K扩充到paddingSize的整数倍，paddingSize默认为4。

- mlir::createLinalgNamedOpConversionPass

  将depth_multiplier=1的`linalg.depthwise_conv_2d_nhwc_hwcm`转换成`linalg.depthwise_conv_2d_nhwc_hwc`，将depth_multiplier=1的`linalg.depthwise_conv_2d_nhwc_hwcm_q`转换成`linalg.depthwise_conv_2d_nhwc_hwc_q`。

  depth_multiplier的作用见 https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D 。

  ```txt
  The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.
  ```

- IREE::Flow::createExpandTensorShapesPass

  将dynamic tensor扩充为tensor + dynamic dim的对偶形式，这么做的一个好处是动态维度可以直接参与计算和推导。比如

  ```c++
    // func.func private @add(%arg0 : tensor<?x2xf32>, %arg1 : tensor<?x2xf32>) -> tensor<?x2xf32>
    // iree_input.global private mutable @param : tensor<?x2xf32>
    // func.func @run(%arg0 : tensor<?x2xf32>) -> tensor<?x2xf32> {
    //   %0 = iree_input.global.load @param : tensor<?x2xf32>
    //   %1 = call @add(%0, %arg0) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
    //   iree_input.global.store %1, @param : tensor<?x2xf32>
    //   return %1 : tensor<?x2xf32>
    // }
    func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
    util.global private mutable @param : tensor<?x2xf32>
    func.func @run(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %c0 = arith.constant 0 : index
      %param = util.global.load @param : tensor<?x2xf32>
      %dim = tensor.dim %param, %c0 : tensor<?x2xf32>
      %0 = hal.tensor.export %param : tensor<?x2xf32>{%dim} -> !hal.buffer_view
      %1 = call @add(%0, %arg0) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
      %2 = hal.buffer_view.dim<%1 : !hal.buffer_view>[0] : index
      %3 = hal.tensor.import %1 : !hal.buffer_view -> tensor<?x2xf32>{%2}
      util.global.store %3, @param : tensor<?x2xf32>
      return %1 : !hal.buffer_view
    }
  ```

  被转换成，

  ```c++
    func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
    util.global private mutable @param : tensor<?x2xf32>
    util.global private mutable @param__d0 : index
    func.func @run(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %c0 = arith.constant 0 : index
      %param = util.global.load @param : tensor<?x2xf32>
      %param__d0 = util.global.load @param__d0 : index
      %0 = flow.tensor.tie_shape %param : tensor<?x2xf32>{%param__d0}
      %dim = tensor.dim %0, %c0 : tensor<?x2xf32>
      %1 = hal.tensor.export %0 : tensor<?x2xf32>{%dim} -> !hal.buffer_view
      %2 = call @add(%1, %arg0) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
      %3 = hal.buffer_view.dim<%2 : !hal.buffer_view>[0] : index
      %4 = hal.tensor.import %2 : !hal.buffer_view -> tensor<?x2xf32>{%3}
      util.global.store %4, @param : tensor<?x2xf32>
      util.global.store %3, @param__d0 : index
      return %2 : !hal.buffer_view
    }
  ```

  从中可以看出几点变化：

  - global tensor增加了一个表示动态维度的global index。

    ```c++
    util.global private mutable @param : tensor<?x2xf32>
    
    转换成：
    util.global private mutable @param : tensor<?x2xf32>
    util.global private mutable @param__d0 : index
    ```

  - global load

    ```c++
    %param = util.global.load @param : tensor<?x2xf32>
    
    转换成：
    %param = util.global.load @param : tensor<?x2xf32>
    %param__d0 = util.global.load @param__d0 : index
    %0 = flow.tensor.tie_shape %param : tensor<?x2xf32>{%param__d0}
    ```

  - global store

    ```c++
    util.global.store %3, @param : tensor<?x2xf32>
    
    转换成：
    util.global.store %4, @param : tensor<?x2xf32>
    util.global.store %3, @param__d0 : index
    ```

- buildGlobalOptimizationPassPipeline

  - IREE::Util::createSimplifyGlobalAccessesPass

    这个pass主要做这几件事：

    - 将不可变global tensor的load提前到了block的开头，将global tensor的store安全地挪到block的结尾。

    - 进行以下化简：

      - 如果load after store，则把load直接替换成store的source。比如，

        ```c++
        store %0, @p
        %1 = load @p
        return %1
        ```

        会被转换成，

        ```c++
        store %0, @p
        return %0
        ```

      - 如果store after store，则直接消除前一个store

        ```c++
        store %0, @p
        store %1, @p
        ```

        会被转换成，

        ```c++
        store %1, @p
        ```

      - 如果load after load，则消除后一个load

        ```c++
        %0 = load @p
        %1 = load @p
        return %1
        ```

        会被转换成，

        ```c++
        %0 = load @p
        return %0
        ```

    一个完整的例子：

    ```c++
      func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
      util.global private mutable @param0 : tensor<1x2xf32>
      util.global private @param1 : tensor<1x2xf32>
      func.func @run(%arg0: !hal.buffer_view) attributes {iree.abi.stub} {
        %param0 = util.global.load @param0 : tensor<1x2xf32>
        %0 = hal.tensor.export %param0 : tensor<1x2xf32> -> !hal.buffer_view
        %1 = call @add(%0, %arg0) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
        %2 = hal.tensor.import %1 : !hal.buffer_view -> tensor<1x2xf32>
        util.global.store %2, @param0 : tensor<1x2xf32>
        %param0_0 = util.global.load @param0 : tensor<1x2xf32>
        %param1 = util.global.load @param1 : tensor<1x2xf32>
        %3 = hal.tensor.export %param0_0 : tensor<1x2xf32> -> !hal.buffer_view
        %4 = hal.tensor.export %param1 : tensor<1x2xf32> -> !hal.buffer_view
        %5 = call @add(%3, %4) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
        %6 = hal.tensor.import %5 : !hal.buffer_view -> tensor<1x2xf32>
        util.global.store %6, @param0 : tensor<1x2xf32>
        return
      }
    ```

    转换成，

    ```c++
    func.func private @add(!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub}
      util.global private mutable @param0 : tensor<1x2xf32>
      util.global private @param1 : tensor<1x2xf32>
      func.func @run(%arg0: !hal.buffer_view) attributes {iree.abi.stub} {
        %param0 = util.global.load @param0 : tensor<1x2xf32>
        %param1 = util.global.load @param1 : tensor<1x2xf32>
        %0 = hal.tensor.export %param0 : tensor<1x2xf32> -> !hal.buffer_view
        %1 = call @add(%0, %arg0) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
        %2 = hal.tensor.import %1 : !hal.buffer_view -> tensor<1x2xf32>
        %3 = hal.tensor.export %2 : tensor<1x2xf32> -> !hal.buffer_view
        %4 = hal.tensor.export %param1 : tensor<1x2xf32> -> !hal.buffer_view
        util.global.store %2, @param0 : tensor<1x2xf32>
        %5 = call @add(%3, %4) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
        %6 = hal.tensor.import %5 : !hal.buffer_view -> tensor<1x2xf32>
        util.global.store %6, @param0 : tensor<1x2xf32>
        return
      }
    ```

    这个例子中将param1的load操作提前，并且将`%param0_0 = util.global.load @param0 : tensor<1x2xf32>`直接替换为`%2`。

  - IREE::Util::createApplyPatternsPass

    执行IREE::Util dialect ODS中定义的Canonicalization Patterns，并执行block和跳转命令参数化简操作。

    - block参数化简

      ```c++
      br ^bb1(%0, %0 : index, index)
      ^bb1(%arg0: index, %arg1: index):
        ...
      ```

      折叠相同的参数，化简为

      ```c++
      br ^bb1(%0 : index)
      ^bb1(%arg0: index):  // %arg1 remapped to %arg0
        ...
      ```

    - 跳转命令参数消除

      ```c++
      func.func @foo(%arg0: index) {
        br ^bb1(%arg0 : index)
        ^bb1(%0: index):
          ...
      }
      ```

      消除参数后，

      ```c++
      func.func @foo(%arg0: index) {
        br ^bb1
        ^bb1:  // %0 remapped to %arg0
          ...
      }
      ```

  - IREE::Util::createFoldGlobalsPass

    这个pass继续对global tensor的load和store操作进行优化，主要包括：

    - 内联常量store，比如

      ```c++
      util.global mutable @a : i32
      func.func @fool {
        %c5 = arith.constant 5 : i32
        util.global.store %c5, @a : i32
        return
      }
      ```

      转换成，

      ```c++
      util.global @a = 5 : i32
      ```

    - 內联常量load，比如

      ```c++
      util.global @a = 5 : i32
      func.func @fool {
        %1 = util.global.load @a : i32
        ...
      }
      ```

      转换成，

      ```c++
      func.func @fool {
        %1 = arith.constant 5 : i32
        ...
      }
      ```

    - 重命名互为链式的global tensor。
    - 如果一个mutable global tensor只在init函数中被store过，则将它修改为immutable。
    - 删除没有load过的global tensor。
    - 合并相同初始值的immutable global tensor。

  - IREE::Util::createHoistIntoGlobalsPass

- IREE::Flow::createTensorPadToTensorInsertSlicePass

  将`tensor.pad`转换为`linalg.fill` + `tensor.insert_slice`。

  ```c++
    func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x1xf32>
      %padded = tensor.pad %0 low[1, 2] high[3, 4] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %cst : f32
      } : tensor<1x1xf32> to tensor<5x7xf32>
      %1 = hal.tensor.export %padded : tensor<5x7xf32> -> !hal.buffer_view
      return %1 : !hal.buffer_view
    }
  ```

  转换为，

  ```c++
    func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x1xf32>
      %1 = tensor.empty() : tensor<5x7xf32>
      %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<5x7xf32>) -> tensor<5x7xf32>
      %inserted_slice = tensor.insert_slice %0 into %2[1, 2] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<5x7xf32>
      %3 = hal.tensor.export %inserted_slice : tensor<5x7xf32> -> !hal.buffer_view
      return %3 : !hal.buffer_view
    }
  ```

- mlir::createConvertElementwiseToLinalgPass

  把elementwise算子（带有Elementwise traits的op）转换成linalg generic op，方便后续对elementwise op做算子融合。arith dialect和math dialect的op都是Elementwise的，所以实际上这个pass会把arith dialect和math dialect lower到linalg dialect。

  ```c++
  func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x3xf32>
    %1 = arith.addf %0, %0 : tensor<2x3xf32>
    %2 = hal.tensor.export %1 : tensor<2x3xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
  ```

  转换成，

  ```c++
  func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    } -> tensor<2x3xf32>
    %2 = hal.tensor.export %1 : tensor<2x3xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
  ```

- mlir::createLinalgFoldUnitExtentDimsPass

  消除长度为1的维度或者循环。

  ```c++
  func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x3xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %in : f32
      linalg.yield %3 : f32
    } -> tensor<1x3xf32>
    %2 = hal.tensor.export %1 : tensor<1x3xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
  ```

  转换成，

  ```c++
  func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<1x3xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<1x3xf32> into tensor<3xf32>
    %collapsed_0 = tensor.collapse_shape %0 [[0, 1]] : tensor<1x3xf32> into tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%collapsed : tensor<3xf32>) outs(%collapsed_0 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %in : f32
      linalg.yield %3 : f32
    } -> tensor<3xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<3xf32> into tensor<1x3xf32>
    %2 = hal.tensor.export %expanded : tensor<1x3xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
  ```

  可以看到其中的`linalg.generic`由2层循环缩减成了单层循环。

- createInterchangeGenericOpsPass

  循环维度变换。将reduction循环维度交换到最内层，相应的parallel循环维度被交换到外层。

  ```c++
  // sum(%arg0: tensor<2x3xf32>, 0) -> tensor<3xf32>
  func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x3xf32>
    %1 = tensor.empty() : tensor<3xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<3xf32>) -> tensor<3xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%0 : tensor<2x3xf32>) outs(%2 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<3xf32>
    %4 = hal.tensor.export %3 : tensor<3xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
  ```

  交换循环之后转换成，

  ```c++
  func.func @foo(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x3xf32>
    %1 = tensor.empty() : tensor<3xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<3xf32>) -> tensor<3xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<2x3xf32>) outs(%2 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<3xf32>
    %4 = hal.tensor.export %3 : tensor<3xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
  ```

- memref::createResolveShapedTypeResultDimsPass

- mlir::createCanonicalizerPass

- mlir::createCSEPass

- createFusionOfTensorOpsPass

  主要做elementwise的算子融合，其次也会将`tensor.expand_shape`转换成`linalg generic op`，方便进行算子融合。

  elementwise算子融合的条件：

  - producer和comsumer都是linalg generic op，且都为tensor语义。
  - producer只有一个use。
  - producer所有维度的迭代类型都是parallel，consumer的index map必须和producer具有相同的循环嵌套层数。
  - producer结果的index map必须是Permutation，即结果的每个元素有且仅store一次（输出是pointwise的）。
  - consumer可以包含reduction迭代类型，但需要保证融合后输入的index map可以覆盖每一个迭代维度，理由是如果缺失就无法确定该维度的循环边界。

  ```c++
  // reduce(mul(arg0, arg1), 0)
  // for (int d0 = 0; d0 < n; ++d0) {
  //   temp[d0] = arg0[d0] * arg1[d0];
  // }
  // result = 0;
  // for (int d0 = 0; d0 < n; ++d0) {
  //   result += temp[d0];
  // }
  func.func @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2xf32>
    %2 = tensor.empty() : tensor<2xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<2xf32>, tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<2xf32>
    %4 = tensor.empty() : tensor<f32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<f32>) -> tensor<f32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%3 : tensor<2xf32>) outs(%5 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.addf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<f32>
    %7 = hal.tensor.export %6 : tensor<f32> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
  ```

  融合mul和reduce之后转换成，

  ```c++
  // result = 0;
  // for (int d0 = 0; d0 < n; ++d0) {
  //   result += arg0[d0] * arg1[d0];
  // }
  func.func @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%0, %1 : tensor<2xf32>, tensor<2xf32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.mulf %in, %in_0 : f32
      %7 = arith.addf %6, %out : f32
      linalg.yield %7 : f32
    } -> tensor<f32>
    %5 = hal.tensor.export %4 : tensor<f32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
  ```

- mlir::createLinalgDetensorizePass

  将0-D Tensor转换为它的基础元素类型。

- mlir::createCanonicalizerPass

- mlir::createCSEPass

- createSplitReductionPass

  将matmul和topk的单次reduce分成两次reduce操作（一次batch matmul和一次add）。默认不开启，设置--iree-flow-split-matmul-reduction选项>=2可开启。

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<128x256xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<256x256xf32>
    %2 = linalg.init_tensor [128, 256] : tensor<128x256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<128x256xf32>, tensor<256x256xf32>) outs(%3 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %5 = hal.tensor.export %4 : tensor<128x256xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
  ```

  --iree-flow-split-matmul-reduction=2转换成，

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<128x256xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<256x256xf32>
    %2 = linalg.init_tensor [128, 256] : tensor<128x256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %4 = tensor.expand_shape %0 [[0], [1, 2]] : tensor<128x256xf32> into tensor<128x2x128xf32>
    %5 = tensor.expand_shape %1 [[0, 1], [2]] : tensor<256x256xf32> into tensor<2x128x256xf32>
    %6 = linalg.init_tensor [2, 128, 256] : tensor<2x128x256xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4, %5 : tensor<128x2x128xf32>, tensor<2x128x256xf32>) outs(%7 : tensor<2x128x256xf32>) attrs =  {__internal_linalg_transform__ = "SPLIT", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %11 = arith.mulf %arg2, %arg3 : f32
      %12 = arith.addf %arg4, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<2x128x256xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%8 : tensor<2x128x256xf32>) outs(%3 : tensor<128x256xf32>) attrs =  {__internal_linalg_transform__ = "SPLIT"} {
    ^bb0(%arg2: f32, %arg3: f32):
      %11 = arith.addf %arg2, %arg3 : f32
      linalg.yield %11 : f32
    } -> tensor<128x256xf32>
    %10 = hal.tensor.export %9 : tensor<128x256xf32> -> !hal.buffer_view
    return %10 : !hal.buffer_view
  }
  ```

- createInterchangeGenericOpsPass

  循环维度变换。将reduction循环维度交换到最内层，相应的parallel循环维度被交换到外层。

- createInterchangeTransposeGenericOpsPass

  当输入indexing map是permutation时，交换循环维度使得输入的indexing map是identity的，其作用是使得输入尽可能变成连续访存。

- createDispatchWithTransformDialect

  根据transform dialect对算子进行调度和派遣，需要另外加载一个transform dialect的module文件，默认不做该变换。transform dialect定义了一套调度规则，用于引导目标IR进行变换，比如循环展开、tiling等。

- createFormDispatchRegionsPass

  以包含reduction loop的linalg op或named linalg op为中心（root），按一定规则合并producers和comsumers，划分出dispatch region子图。dispatch region是IREE中的原子执行单元，dispatch region内部可以直接复用输入和输出的内存，从而避免了内部的内存分配操作，内存分配只发生在dispatch region的边界，同时dispatch region之间会自动插入同步操作。

  ```c++
  func.func @predict(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x10xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<10x5xf32>
    %2 = hal.tensor.import %arg2 : !hal.buffer_view -> tensor<5xf32>
    %3 = tensor.empty() : tensor<2x5xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x5xf32>) -> tensor<2x5xf32>
    %5 = linalg.matmul ins(%0, %1 : tensor<2x10xf32>, tensor<10x5xf32>) outs(%4 : tensor<2x5xf32>) -> tensor<2x5xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<2x5xf32>, tensor<5xf32>) outs(%3 : tensor<2x5xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<2x5xf32>
    %7 = hal.tensor.export %6 : tensor<2x5xf32> -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
  ```

  转换成，

  ```c++
  func.func @predict(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x10xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<10x5xf32>
    %2 = hal.tensor.import %arg2 : !hal.buffer_view -> tensor<5xf32>
    %3 = tensor.empty() : tensor<2x5xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x5xf32>) -> tensor<2x5xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_0 = arith.constant 1 : index
    %5 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>()[%c0, %c2, %c1_0]
    %c0_1 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1_2 = arith.constant 1 : index
    %6 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>()[%c0_1, %c5, %c1_2]
    %7 = flow.dispatch.region[%5, %6] -> (tensor<2x5xf32>) {
      %9 = linalg.matmul ins(%0, %1 : tensor<2x10xf32>, tensor<10x5xf32>) outs(%4 : tensor<2x5xf32>) -> tensor<2x5xf32>
      %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %2 : tensor<2x5xf32>, tensor<5xf32>) outs(%3 : tensor<2x5xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %11 = arith.addf %in, %in_3 : f32
        linalg.yield %11 : f32
      } -> tensor<2x5xf32>
      flow.return %10 : tensor<2x5xf32>
    } count(%arg3: index, %arg4: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg3, %arg4
      flow.return %x, %y, %z : index, index, index
    }
    %8 = hal.tensor.export %7 : tensor<2x5xf32> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
  ```

- createFormDispatchWorkgroupsPass

  将dispatch region转换成dispatch work group的形式，并将cloneable的op（比如`tensor.fill`、`tensor.empty`等）拷贝到work group中。如果在linalg层做了tiling，该pass也会把tiling引入的`tensor.extract_slice`和`tensor.insert_slice`尽可能转换成`flow.tensor.slice`和`flow.tensor.update`，转换不了的后续再转换成`flow.dispatch.tensor.load`和`flow.dispatch.tensor.store`。这里上一步的结果会被转换成，

  ```c++
  func.func @predict(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2x10xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<10x5xf32>
    %2 = hal.tensor.import %arg2 : !hal.buffer_view -> tensor<5xf32>
    %3 = flow.dispatch.workgroups[%c2, %c5](%0, %1, %2) : (tensor<2x10xf32>, tensor<10x5xf32>, tensor<5xf32>) -> tensor<2x5xf32> =
        (%arg3: !flow.dispatch.tensor<readonly:tensor<2x10xf32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<10x5xf32>>, %arg5: !flow.dispatch.tensor<readonly:tensor<5xf32>>, %arg6: !flow.dispatch.tensor<writeonly:tensor<2x5xf32>>) {
      %cst = arith.constant 0.000000e+00 : f32
      %5 = flow.dispatch.tensor.load %arg3, offsets = [0, 0], sizes = [2, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10xf32>> -> tensor<2x10xf32>
      %6 = flow.dispatch.tensor.load %arg4, offsets = [0, 0], sizes = [10, 5], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10x5xf32>> -> tensor<10x5xf32>
      %7 = flow.dispatch.tensor.load %arg5, offsets = [0], sizes = [5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<5xf32>> -> tensor<5xf32>
      %8 = tensor.empty() : tensor<2x5xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<2x5xf32>) -> tensor<2x5xf32>
      %10 = linalg.matmul ins(%5, %6 : tensor<2x10xf32>, tensor<10x5xf32>) outs(%9 : tensor<2x5xf32>) -> tensor<2x5xf32>
      %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %7 : tensor<2x5xf32>, tensor<5xf32>) outs(%8 : tensor<2x5xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %12 = arith.addf %in, %in_0 : f32
        linalg.yield %12 : f32
      } -> tensor<2x5xf32>
      flow.dispatch.tensor.store %11, %arg6, offsets = [0, 0], sizes = [2, 5], strides = [1, 1] : tensor<2x5xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x5xf32>>
      flow.return
    } count(%arg3: index, %arg4: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg3, %arg4
      flow.return %x, %y, %z : index, index, index
    }
    %4 = hal.tensor.export %3 : tensor<2x5xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
  ```

- createCaptureDispatchDynamicDimsPass

  由于`flow.dispatch.workgroups`的参数中动态形状tensor被替换成了`!flow.dispatch.tensor`和相应的动态维度index，该pass捕获workgroups参数中的动态维度index，插入`flow.dispatch.tie_shape`将参数中的动态维度index和`!flow.dispatch.tensor`进行绑定。

  ```c++
  // func.func @test(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  //  %0 = mhlo.add %arg0, %arg1 : tensor<?xf32>
  //  return %0 : tensor<?xf32>
  // }
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %3 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<?xf32>{%2}
    %4 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>()[%c0, %0, %c1]
    %5 = flow.dispatch.workgroups[%4](%0, %1, %3, %0, %2, %0) : (index, tensor<?xf32>{%0}, tensor<?xf32>{%2}, index, index, index) -> tensor<?xf32>{%0} =
        (%arg2: index, %arg3: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: !flow.dispatch.tensor<writeonly:tensor<?xf32>>) {
      %7 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [%arg7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg7} -> tensor<?xf32>
      %8 = flow.dispatch.tensor.load %arg4, offsets = [0], sizes = [%arg6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg6} -> tensor<?xf32>
      %9 = tensor.empty(%arg7) : tensor<?xf32>
      %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7, %8 : tensor<?xf32>, tensor<?xf32>) outs(%9 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %11 = arith.addf %in, %in_0 : f32
        linalg.yield %11 : f32
      } -> tensor<?xf32>
      flow.dispatch.tensor.store %10, %arg8, offsets = [0], sizes = [%arg7], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg7}
      flow.return
    } count(%arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2
      flow.return %x, %y, %z : index, index, index
    }
    %6 = hal.tensor.export %5 : tensor<?xf32>{%0} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
  ```

  会被转换成，

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %3 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<?xf32>{%2}
    %4 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>()[%c0, %0, %c1]
    %5 = flow.dispatch.workgroups[%4](%0, %1, %3, %0, %2, %0) : (index, tensor<?xf32>{%0}, tensor<?xf32>{%2}, index, index, index) -> tensor<?xf32>{%0} =
        (%arg2: index, %arg3: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: !flow.dispatch.tensor<writeonly:tensor<?xf32>>) {
      %7 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg7}
      %8 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg6}
      %9 = flow.dispatch.tie_shape %arg8 : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg7}
      %10 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [%arg7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg7} -> tensor<?xf32>
      %11 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [%arg6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg6} -> tensor<?xf32>
      %12 = tensor.empty(%arg7) : tensor<?xf32>
      %13 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %11 : tensor<?xf32>, tensor<?xf32>) outs(%12 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %14 = arith.addf %in, %in_0 : f32
        linalg.yield %14 : f32
      } -> tensor<?xf32>
      flow.dispatch.tensor.store %13, %9, offsets = [0], sizes = [%arg7], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg7}
      flow.return
    } count(%arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2
      flow.return %x, %y, %z : index, index, index
    }
    %6 = hal.tensor.export %5 : tensor<?xf32>{%0} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
  ```

- mlir::createCanonicalizerPass

- createCSEPass

- createInitializeEmptyTensorsPass

  如果`tensor.empty` op的user中存在非linalg或IREE LinalgExt op，则把该`tensor.empty` op转换成`flow.tensor.empty`或`flow.tensor.splat` op。

- IREE::Flow::createOutlineDispatchRegionsPass

  把每个dispatch region转换成`flow.executable` + `flow.dispatch` op。

  ```c++
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c2 = arith.constant 2 : index
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2xf32>
    %2 = flow.dispatch.workgroups[%c2](%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> =
          (%arg2: !flow.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: !flow.dispatch.tensor<readonly:tensor<2xf32>>, %arg4: !flow.dispatch.tensor<writeonly:tensor<2xf32>>) {
      %4 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
      %5 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
      %6 = tensor.empty() : tensor<2xf32>
      %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<2xf32>, tensor<2xf32>) outs(%6 : tensor<2xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %8 = arith.addf %in, %in_0 : f32
        linalg.yield %8 : f32
      } -> tensor<2xf32>
      flow.dispatch.tensor.store %7, %arg4, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
      flow.return
    } count(%arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2
      flow.return %x, %y, %z : index, index, index
    }
    %3 = hal.tensor.export %2 : tensor<2xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
  ```

  转换成

  ```c++
  flow.executable private @test_dispatch_0 {
    flow.executable.export public @test_dispatch_0_generic_2 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
      flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
      func.func @test_dispatch_0_generic_2(%arg0: !flow.dispatch.tensor<readonly:tensor<2xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<2xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<2xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %2 = tensor.empty() : tensor<2xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<2xf32>, tensor<2xf32>) outs(%2 : tensor<2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %4 = arith.addf %in, %in_0 : f32
          linalg.yield %4 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        return
      }
    }
  }
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c2 = arith.constant 2 : index
    %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<2xf32>
    %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<2xf32>
    %2 = flow.dispatch @test_dispatch_0::@test_dispatch_0_generic_2[%c2](%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %3 = hal.tensor.export %2 : tensor<2xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
  ```

- IREE::Util::createStripDebugOpsPass

  消除DebugOnly op。

- mlir::createCanonicalizerPass

- IREE::Flow::createDeduplicateExecutablesPass

  消除重复的`flow.executable`。

- IREE::Flow::createInjectDispatchTracingPass

  注入跟踪运行时dispatch函数输入和输出信息的op。默认不开启。

- IREE::Flow::createCleanupTensorShapesPass

  删除`flow.tensor.tie_shape` op，并确认module中不再包含`tensor.dim`和`tensor.rank` 这两类形状查询op。

- mlir::createCanonicalizerPass

- mlir::createCSEPass

- mlir::createCanonicalizerPass

- mlir::createCSEPass

- mlir::createSymbolDCEPass
