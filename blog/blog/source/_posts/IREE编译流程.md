---
title: IREE编译流程解析(一)

date: 2023-1-4 12:00:04

category: DL Compiler

tags: [IREE, Deep Learning Compiler]

---

IREE支持将MHLO、XLA、Torch Tensor和TOSA作为输入，经过一系列优化passes编译成IREE定义的VM bytecode文件，其中硬件相关代码会编译成相应的Executable，保存在VM bytecode中供host进行调用。比如CUDA相关的计算代码会被lowering成PTX代码，在IREE的runtime中再被CUDA的运行时以JIT的方式编译成可执行的cubin kernel。

<!-- more -->

IREE编译的主要流程是IREEVMTransformPassPipeline，IREEVMTransformPassPipeline又被分成InputConversionPassPipeline、CommonInputConversionPassPipeline、ABI::TransformPassPipeline、Flow::FlowTransformPassPipeline、Stream::StreamTransformPassPipeline（仅CUDA后端）、HAL::HALTransformPassPipeline、VM::VMTransformPassPipeline等几个子阶段。

- InputConversionPassPipeline

  主要作用是将不同的输入（MHLO、XLA、Torch Tensor和TOSA）统一lower成linalg dialect和builtin的arith dialect、scf dialect和tensor dialect。

- CommonInputConversionPassPipeline

  主要作用是将IREE::Input dialect lower成IREE::Util、IREE::Flow和IREE::HAL dialect。

- ABI::TransformPassPipeline

  主要作用是将外部导入的接口和本module导出到外部的接口参数统一成标准标量类型或`hal.buffer_view`类型（`hal.buffer_view`对应tensor）。

- Flow::FlowTransformPassPipeline

  主要作用是执行一系列窥孔优化，比如1x1的conv2d转换成matmul、tiling、op fusion等。

- Stream::StreamTransformPassPipeline

- HAL::HALTransformPassPipeline

- VM::VMTransformPassPipeline
