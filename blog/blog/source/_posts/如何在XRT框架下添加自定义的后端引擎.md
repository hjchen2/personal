---
title: 如何在XRT框架下添加自定义的后端引擎

date: 2020-02-25 16:06:18

category: XRT

tags: [XRT, Compiler, TensorFlow XLA, TensorRT]

---

XRT为不同的后端引擎提供了统一的上层功能和接口抽象，这些功能和接口包括：

- 统一的DAG计算图表示
- 统一的子图表达、切分和折叠过程
- 统一的JIT子图编译接口和缓存机制
- 统一的Executable Launch接口

得益于上层统一的抽象和模块化的设计，后端引擎只需要处理一些差异化的接口，并且这些差异化通常只体现在子图的编译和executable launch接口的具体实现上。

<!-- more -->

我们把XRT的每个子图都看成是一个function，function包含输入和输出参数，以及对应的函数体（DAG表示的计算图），比如下面表示的是只包含一个relu节点的XRT子图，其中node表示计算节点，input和output分别表示子图的输入和输出。

```txt
function {
  input {
    name: "_xrt_entry_0"
    value: "_MyGraph_0_input.0.0_2/out"
  }
  output {
    name: "_xrt_return_0"
    value: "relu-0/y_0"
  }
  node {
    name: "relu-0"
    device_tag: "cuda"
    user_conf {
      op_type_name: "relu"
      input {
        key: "x"
        value {
          s: "_MyGraph_0_input.0.0_2/out"
        }
      }
      output {
        key: "y"
        value {
          s: "relu-0/y_0"
        }
      }
    }
  }
}
```

在runtime阶段function首先需要被编译成executable，执行function实际上就是feed对应的输入参数去launch这个编译好的executable，同时得到执行的结果，即function的返回值。

在XRT框架下每个后端引擎都有一个与之相对应的executable（比如XLA的XlaExecutable和TensorRT的TrtExecutable），和将function编译成对应executable的compiler（比如XLA的XlaGraphCompiler和TensorRT的TrtGraphCompiler），因此添加一个新的后端引擎，通常只需要添加一个对应的executable和compiler。下面以添加一个自定义的后端引擎Toy为例，详细介绍在XRT框架下支持新的后端引擎的具体过程。

首先在xrt.proto文件中XrtEngine下增加一个Toy引擎字段。

```c++
enum XrtEngine {
  DEFAULT = 1;
  XLA = 2;
  TENSORRT = 3;
  TVM = 4;
  TOY = 5;  // For Toy engine
}
```

如果Toy引擎针对的硬件不在XrtDevice中，则需要在XrtDevice中增加对应的设备字段。这里我们假设自定义的Toy引擎只支持GPU_CUDA，因此就不需要修改XrtDevice了。

接下来，与XLA和TensorRT一样，我们在`oneflow_xrt/compiler`目录下创建一个toy目录，其余所有与Toy引擎相关的代码都将放在该目录下。

## Toy Executable

在增加任何一个后端引擎之前，我们都需要仔细考虑该后端引擎所需的最小执行环境，一个最简单的执行环境包括输入输出、中间结果以及执行具体计算逻辑的硬件代码，这个代码可以是通过codegen自动生成的，也可以是手工实现的。

接下来我们给自定义的Toy引擎增加一个对应的ToyExecutable。在`oneflow_xrt/compiler/toy`目录下，我们创建文件toy_executable.h和toy_executable.cpp。

toy_executable.h中定义ToyExecutable，ToyExecutable必须继承自Executable，并实现Run接口。为了尽可能简单，ToyExecutable只包含输出outputs、中间结果tmp_buffers和编排好的函数调用列表func_codes，以及每个函数的输入输出参数对应的buffer序号func_args_。

```c++
#ifndef ONEFLOW_XRT_COMPILER_TOY_TOY_EXECUTABLE_H_

#include "oneflow_xrt/compiler/executable.h"
#include "oneflow_xrt/compiler/parameter.h"

#include <vector>
#include <functional>

namespace oneflow {
namespace xrt {

typedef std::function<void(const std::vector<Parameter> &,
                           const std::vector<Parameter> &)> FuncCode;

struct FuncArgumentIndices {
  std::vector<int> inputs;
  std::vector<int> outputs;
};

class ToyExecutable : public Executable {
 public:
  ToyExecutable(const std::string &name, const int num_inputs,
                 const std::vector<Parameter> &outputs,
                 const std::vector<Parameter> &temp_buffers,
                 const std::vector<FuncCode> &func_codes,
                 const std::vector<FuncArgumentIndices> &func_args);

  bool Run(const std::vector<Parameter> &inputs,
           const ExecutableRunOptions &run_options,
           bool block_until_done = true) override;

 private:
  int num_inputs_;
  std::vector<Parameter> outputs_;
  std::vector<Parameter> temp_buffers_;
  std::vector<FuncCode> func_codes_;
  std::vector<FuncArgumentIndices> func_args_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMPILER_TOY_TOY_EXECUTABLE_H_
```

在toy_executable.cpp中实现Run方法，这里我们只是简单的顺序执行编排好的函数func_codes。

```c++
#include "oneflow_xrt/compiler/toy/toy_executable.h"

namespace oneflow {
namespace xrt {

ToyExecutable::ToyExecutable(const std::string &name, const int num_inputs,
                               const std::vector<Parameter> &outputs,
                               const std::vector<Parameter> &temp_buffers,
                               const std::vector<FuncCode> &func_codes,
                               const std::vector<FuncArgumentIndices> &func_args)
    : Executable(name, XrtEngine::TOY),
      num_inputs_(num_inputs),
      outputs_(outputs),
      temp_buffers_(temp_buffers),
      func_codes_(func_codes),
      func_args_(func_args) {}

bool ToyExecutable::Run(const std::vector<Parameter> &inputs,
                         const ExecutableRunOptions &run_options,
                         bool block_until_done) {
  auto PullArgs = [&](const std::vector<int> &indices) {
    std::vector<Parameter> args;
    for (int idx : indices) {
      if (idx < num_inputs_) {
        args.push_back(inputs[idx]);
      } else if (idx < num_inputs_ + outputs_.size()) {
        args.push_back(outputs_[idx - num_inputs_]);
      } else {
        idx -= (num_inputs_ + outputs_.size());
        CHECK_GE(idx, 0);
        CHECK_LT(idx, temp_buffers_.size());
        args.push_back(temp_buffers_[idx]);
      }
    }
    return std::move(args);
  };

  CHECK_EQ(inputs.size(), num_inputs_);

  for (int i = 0; i < func_codes_.size(); ++i) {
    auto in_args = PullArgs(func_args_[i].inputs);
    auto out_args = PullArgs(func_args_[i].outputs);
    func_codes_[i](in_args, out_args);
  }

  // Synchronize stream if block_until_done
  if (block_until_done) {
    // TODO()
  }

  // All return params are the results of the executable
  this->results_ = run_options.return_params;
  return true /*running status*/;
}

}  // namespace xrt
}  // namespace oneflow
```

目前为止我们已经完成了一个最简单的运行时executable，这个executable甚至有点类似其他框架中提供的最简单的图执行器（graph executor）。接下来我们要介绍如何将一个XRT的子图编译成上面的ToyExecutable。

## Toy Compiler

每个后端引擎都对应一个compiler，当我们希望使用某个后端引擎来执行一个XRT子图时，就需要有一个对应的compiler将该子图编译成后端引擎对应的executable。Compiler通常都非常注重编译产物的执行性能，而性能以外的关切点也导致了不同的技术路线，比如对算法通用性、跨平台有高度关切的TVM和XLA采用了LLVM传统编译器的路线，而对于过分看重性能但硬件平台单一的TensorRT更多的则是采用手工优化和tuning相结合的策略。不过这两种技术路线并不是完全对立的，也是在不断地相互借鉴和融合。

在XRT中，所有这些技术方案都是可以被兼容的，你可以根据实际情况自由切换，你也可以把XRT当成实验场所，实现一个自定义的compiler，并在同一套框架下对比不同compiler、不同技术方案的优劣。

回到本文的主题，我们现在需要实现一个ToyExecutable对应的compiler，我们也把该compiler叫做ToyGraphCompiler。

首先在`oneflow_xrt/compiler/toy`目录下新建两个文件toy_graph_compiler.h和toy_graph_compiler.cpp。在toy_graph_compiler.h文件中定义类ToyGraphCompiler，ToyGraphCompiler必须继承自类GraphCompiler::Impl，并实现对应的Compile接口。

```c++
class ToyGraphCompiler : public GraphCompiler::Impl {
 public:
  explicit ToyGraphCompiler(const std::string &name)
      : GraphCompiler::Impl(name) {}

  virtual ~ToyGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(
      const XrtGraph *graph,
      const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) override;
};
```

在toy_graph_compiler.cpp中实现Compile接口，并注册一个新的graph compiler。在动手实现该接口之前，有必要先解释一下该接口的参数列表，graph表示的是function子图，entry_params表示子图的输入，return_params表示子图的输出，aliases通常在包含模型更新操作时会用到，表明输出和输入是一对别名关系。被alias的输入将生命期延长到了整个子图，并且与对应的输出共享内存，因此也就间接实现了inplace计算的目的。

我们按拓扑顺序遍历子图中的每个节点（或op），依次将节点编译成具体的执行代码，并在合适的位置插入临时buffer。为了方便处理不同类型的op，我们在下面的代码中引入了ToyOpContext和ToyOpKernel的概念。

```c++
// Register a new graph compiler for TOY engine.
REGISTER_GRAPH_COMPILER(XrtEngine::TOY, ToyGraphCompiler);

// Realize Compile interface.
std::shared_ptr<Executable> ToyGraphCompiler::Compile(
    const XrtGraph *graph,
    const std::vector<Parameter> &entry_params,
    const std::vector<Parameter> &return_params,
    const std::vector<InputOutputAlias> &aliases) {
  std::vector<Parameter> temp_buffers;
  std::vector<FuncCode> func_codes;
  std::vector<FuncArgumentIndices> func_args;

  std::unordered_map<std::string, int> indices;
  std::unordered_map<std::string, Parameter> all_params;
  for (auto param : entry_params) {
    indices.emplace(param.name(), indices.size());
    all_params[param.name()] = param;
  }
  for (auto param : return_params) {
    indices.emplace(param.name(), indices.size());
    all_params[param.name()] = param;
  }

  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    if (node->IsNoOpNode()) {
      // NoOp node is not computation node, so skip it
      return;
    }

    ToyOpContext op_context(node, all_params);
    auto op_kernel = BuildToyOpKernel(node->type());
    op_kernel->Compile(&op_context);

    func_codes.push_back(op_context.func_code_);

    const auto &buffers = op_context.tmp_buffers_;
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      all_params[it->first] = it->second;
      temp_buffers.push_back(it->second);
      indices.emplace(it->first, indices.size());
    }

    // Finalize argument indices for each function
    FuncArgumentIndices arg_indices;
    for (const auto &arg : op_context.input_args_) {
      arg_indices.inputs.push_back(indices.at(arg));
    }
    for (const auto &arg : op_context.output_args_) {
      arg_indices.outputs.push_back(indices.at(arg));
    }
    func_args.push_back(std::move(arg_indices));
  });

  return std::make_shared<ToyExecutable>(this->name_, entry_params.size(),
                                          return_params, temp_buffers,
                                          func_codes, func_args);
}
```

ToyOpContext临时存储编译需要的元信息和编译结果，为ToyOpKernel提供必要的接口，ToyOpKernel则根据op类型完成单个op的编译过程。上述代码中我们实现了一个将XRT子图编译成ToyExecutable的最简单的graph compiler，下面我们将以ReLU op为例，介绍ToyOpContext和ToyOpKernel是如何对op进行编译的。

## Toy Kernels

我们回过头再仔细研究一下ToyGraphCompiler的Compile实现，ToyOpContext接受两个输入，node和当前所有已经创建过的parameters，经过OpKernel编译后输出函数代码（func_code\_）、中间buffer（tmp_buffers\_），以及函数代码输入和输出对应的parameter names。因此在这个例子中，ToyOpContext被设计成如下形式：

```c++
class ToyOpContext {
 public:
  ToyOpContext(const XrtNode *node,
                const std::unordered_map<std::string, Parameter> &all_params)
      : node_(node), all_params_(all_params) {}

 public:
  const XrtNode *node_;
  const std::unordered_map<std::string, Parameter> &all_params_;

  std::function<void(const std::vector<Parameter>&,
                     const std::vector<Parameter>&)> func_code_;
  std::vector<std::string> input_args_;
  std::vector<std::string> output_args_;
  std::unordered_map<std::string, Parameter> tmp_buffers_;
};
```

对于ToyOpKernel，为了处理不同类型的op，我们采用工厂注册模式，并且这种模式还有另一个用处，就是在XRT划分子图时可以用来判断该引擎是否支持某个类型的op。XRT已经将kernel注册接口封装成了一个辅助类OpKernelRegistrar，但同时也要求ToyOpKernel必须继承基类OpKernel。

```c++
class ToyOpKernel : public OpKernel<ToyOpContext> {
 public:
  virtual void Compile(ToyOpContext *ctx) = 0;
};
```

使用OpKernelRegistrar定义一个用来注册ToyOpKernel的宏。

```c++
#define REGISTER_TOY_OP_KERNEL(OpName, KernelType)                  \
  static auto _toy_op_kernel_##OpName##_ __attribute__((unused)) =  \
      OpKernelRegistrar(#OpName)                                    \
          .SetEngine(XrtEngine::TOY)                                \
          .SetDevice({XrtDevice::GPU_CUDA})                         \
          .SetFactory([]() -> OpKernelBase * {                      \
                        return new KernelType;                      \
                      })
```

最后我们实现一个Relu的OpKernel，填充ToyOpContext的func_code\_、tmp_buffers\_以及输入输出arguments。

```c++
void ComputeRelu(const Parameter &input, const Parameter &output) {
  //TODO(hjchen2)
}

class ToyReluOpKernel : public ToyOpKernel {
 public:
  void Compile(ToyOpContext *ctx) override {
    ctx->func_code_ = [](const std::vector<Parameter> &inputs,
                         const std::vector<Parameter> &outputs) {
      CHECK_EQ(inputs.size(), 1);
      CHECK_EQ(outputs.size(), 1);
      ComputeRelu(inputs[0], outputs[0]);
    };

    for (const XrtEdge *edge : ctx->node_->in_edges()) {
      const auto &name = edge->argument().name();
      CHECK_GT(ctx->all_params_.count(name), 0);
      // TODO(): Filter duplicate input
      ctx->input_args_.push_back(name);
    }

    for (const XrtEdge *edge : ctx->node_->out_edges()) {
      const auto &name = edge->argument().name();
      // TODO(): Filter duplicate output
      ctx->output_args_.push_back(name);
      if (ctx->all_params_.count(name) == 0 &&
          ctx->tmp_buffers_.count(name) == 0) {
        auto param = CreateParameter(name /*argument name*/,
                                     edge->argument().shape(),
                                     edge->argument().data_type());
        ctx->tmp_buffers_[name] = std::move(param);
      }
    }
  }
};
```

最后将ToyReluOpKernel注册到Toy引擎对应的OpKernel工厂下。

```c++
REGISTER_TOY_OP_KERNEL(relu, ToyReluOpKernel)
    .EnableTrainPhase()
    .Finalize();
```

EnableTrainPhase表示该op支持训练，OpKernelRegistrar也提供了其他一些接口，比如设置支持的device列表，mutable variables（inplace更新）和是否是model update op（model update op会影响子图划分）。

## CMake编译

在CMakeList.txt中添加一个BUILD_TOY的选项，并在oneflow_xrt/CMakeLists.txt中添加如下toy引擎模块的编译代码，

```cmake
if(BUILD_TOY)
  file(GLOB_RECURSE XRT_TOY_SRCS compiler/toy/*.cpp)
  add_library(oneflow_xrt_toy ${XRT_TOY_SRCS})
  add_dependencies(
      oneflow_xrt_toy
      ${XRT_THIRD_PARTY_LIBRARIES})
  target_link_libraries(
      oneflow_xrt_toy
      oneflow_xrt
      ${XRT_THIRD_PARTY_LIBRARIES})
  target_include_directories(
      oneflow_xrt_toy PRIVATE ${ONEFLOW_INCLUDE_DIR})
endif()
```

之后在oneflow_xrt/python目录中添加导出Python模块的代码toy_stub.cpp，

```c++
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(_oneflow_xrt_toy_internal, m) {}
```

并在oneflow_xrt/python/CMakeLists.txt中增加如下代码，

```cmake
if(BUILD_TOY)
  oneflow_xrt_add_stub(oneflow_xrt_toy toy_stub.cpp)
endif()
```



## 编译和安装Python wheel包

修改setup.py文件，新增一个toy extension的编译，并在build_ext函数中开启BUILD_TOY选项，

```python
setup_extension(
    "oneflow_xrt_toy",
    cmake_args=["-DBUILD_TOY=ON"],
    description=("oneflow_xrt's toy extension"),
)
```

执行命令`python3 setup.py install`完成wheel包的编译和安装，最后执行如下代码测试添加的toy引擎是否可以正常执行，

```python
import oneflow as flow
import oneflow_xrt as flowrt

class ReluGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
    
    def build(self, input):
        return flow.nn.functional.relu(input)

m = flowrt.XRTModule(ReluGraph(), engine="toy")
x = flow.randn(2, 3, device="cuda")
print(m(x))
```
