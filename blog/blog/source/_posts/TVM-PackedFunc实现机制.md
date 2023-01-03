---

title: TVM PackedFunc实现机制

date: 2020-1-10 12:24:08

category: tvm knowledge

tags: [TVM, PackedFunc]

---

## TVM PackedFunc实现

为了便于Python和C\+\+混合编程，TVM使用了统一的PackedFunc机制。PackedFunc可以将C\+\+中的各类函数打包成统一的函数接口，并自动导出到Python模块中进行调用，并且也支持从Python中注册一个函数，并伪装成PackedFunc在C\+\+和Python中调用。

<!-- more -->

<img src="https://github.com/hjchen2/personal/blob/master/blog/tvm/屏幕快照%202020-01-10%2010.55.45.png?raw=true" style="zoom:36%;" />

### 预备知识

#### Python ctypes混合编程

ctypes是Python自带的跨语言函数调用库，ctypes提供了简单的C数据类型，可以将C/C\+\+动态库中的函数包装成Python函数进行调用。

- 导出C\+\+函数

  首先在C\+\+中定义一个全局函数，并编译生成C\+\+动态库。

  ```c++
  // test.h
  extern "C" {
  int add(int a, int b);
  }
  ```

  ```c++
  // test.cc
  #include "test.h"
  int add(int a, int b) {
    return a + b;
  }
  ```

  用ctypes模块在Python中加载生成的动态库（test.so），并调用C\+\+中的函数。

  ```python
  import ctypes

  # Load shared library
  _LIB = ctypes.CDLL("./test.so", ctypes.RTLD_GLOBAL)

  a = ctypes.c_int(1)
  b = ctypes.c_int(2)
  # Call C func in Python
  print(_LIB.add(a, b))
  # Or
  print(_LIB.add(1, 2))
  ```



- 传递Python函数到C\+\+

  ctypes也支持将Python函数转换成C类型的函数，并在C/C\+\+中进行调用。

  ```python
  def add(a, b):
    return a + b
  ```

  Python add有两个参数a和b，返回值类型与a和b的类型一致。在C\+\+中可以为Python add定义一个函数原型 int(int, int)。

  ```c++
  extern "C" {
  typedef int (*PyCFunc)(int, int);
  int call_py_func(PyCFunc f, int a, int b);
  }
  ```

  ```c++
  #include "test.h"
  int call_py_func(PyCFunc f, int a, int b) {
    return f(a, b);
  }
  ```

  使用ctypes将Python函数转换成C function，传入C\+\+中进行调用。

  ```python
  import ctypes

  cfunc = ctypes.CFUNCTYPE(
      ctypes.c_int, # return type
      ctypes.c_int, # arg0 type
      ctypes.c_int  # arg1 type
      )

  f = cfunc(add)
  # CFUNCTYPE is callable in Python
  print(f(5, 1))

  # Call Python func in C
  print(_LIB.call_py_func(f, 5, 1))
  ```



### PackedFunc实现

#### PackedFunc定义

ctypes可以很方便的将C/C\+\+中的函数导出到Python，调用时直接传入对应的参数即可，但如果需要将Python函数导入到C/C\+\+，则需要在C/C\+\+中提前定义好对应的函数原型（比如上面的PyCFunc），并提供对应函数的调用入口（call_py_func）。为了支持更加灵活的函数定义，TVM将不同类型的函数包装成统一的函数原型。

```c++
void(TVMArgs args, TVMRetValue *rv);
```

统一的函数原型被封装成PackedFunc对象，提供通用的调用接口，直接与调用者进行交互。

```c++
class PackedFunc {
 public:
  using FType = std::function<void (TVMArgs args, TVMRetValue* rv)>;
  template<typename... Args>
  inline TVMRetValue operator()(Args&& ...args) const;
  inline void CallPacked(TVMArgs args, TVMRetValue* rv) const;

  private:
  /*! \brief internal container of packed function */
  FType body_;
};
```

当获得一个PackedFunc对象时，我们就可以像调用普通函数一样调用PackedFunc打包的函数。比如：

```c++
PackedFunc f;
// f(1, 2)首先会自动将参数1，2打包成TVMArgs，接着调用CallPacked，CallPacked最终的执行体是body_
TVMRetValue ret = f(1, 2);
```

#### 函数打包

TVM支持对各类函数进行打包，包括一般的函数、类的成员函数以及lamda表达式。

- 函数原型萃取

  萃取函数原型是为了得到函数的参数和返回值类型。TVM中使用decltype和模版结构体function_signature来实现。

  比如定义一个简单的C函数，

  ```c++
  int add(int a, int b) {
    return a + b;
  }
  ```

  接下来就可以使用如下的代码来萃取add的函数原型，

  ```c++
  template <typename R, typename ...Args>
  struct function_signature<R(Args...)> {
    using FType = R(Args...);
  };

  // 萃取add的函数原型
  using FType = function_signature<decltype(add)>::FType;
  ```

  此外只需要特化function_signature就可以支持函数指针和lambda表达式。注意：TVM function_signature不支持普通成员函数的类型萃取，因此TVM需要借助一个辅助function_signature_helper来对lambda表达式类型进行萃取，而我们这里的function_signature支持普通成员函数，因此lambda表达式类型萃取可以通过递归的function_signature来实现。

  ```c++
  // 普通函数指针
  template <typename R, typename ...Args>
  struct function_signature<R(*)(Args...)> {
    using FType = R(Args...);
  };

  // 非const类的成员函数指针
  template <typename T, typename R, typename ...Args>
   struct function_signature<R(T::*)(Args...)> {
     using FType = R(Args...);
  };

  // const类的成员函数指针
  template <typename T, typename R, typename ...Args>
   struct function_signature<R(T::*)(Args...) const> {
     using FType = R(Args...);
  };

  // lambda表达式
  template<typename T>
  struct function_signature {
    using FType = typename function_signature<decltype(&T::operator())>::FType;
  };
  ```

- 函数打包

  一旦萃取到了函数原型，TVM就利用TypedPackedFunc对普通函数或lambda表达式进行打包。TypedPackedFunc只支持对R(Args...)类型的函数打包，所以如果被打包的函数是一个函数指针，则需要创建一个lambda表达式，转换成R(Args...)类型之后再用TypedPackedFunc对创建的lambda表达式进行打包。

  ```c++
  template<typename R, typename ...Args>
  class TypedPackedFunc<R(Args...)> {
   public:
    using TSelf = TypedPackedFunc<R(Args...)>;
    template<typename FLambda,
             typename = typename std::enable_if<
               std::is_convertible<FLambda,
                                   std::function<R(Args...)>
                                   >::value>::type>
    TypedPackedFunc(const FLambda& typed_lambda) {  // NOLINT(*)
      this->AssignTypedLambda(typed_lambda);
    }
    ...
   private:
    ...
    PackedFunc packed_;
  };
  ```

  当被打包的函数用来实例化TypedPackedFunc对象时，会立刻调用AssignTypedLambda将被打包的函数打包成PackedFunc。

  ```c++
  template<typename R, typename ...Args>
  template<typename FType>
  inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
    packed_ = PackedFunc([flambda](const TVMArgs& args, TVMRetValue* rv) {
        detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
      });
  }
  ```

  AssignTypedLambda实际上是将被打包的函数先封装成了一个函数原型为void(const TVMArgs &args, TVMRetValue *rv)的lambda表达式，然后将这个lambda表达式作为PackedFunc对象的一个成员，通过设置合适的接口（重载operator ()），使得PackedFunc与被打包的源函数表现的完全一样了。

### 自动导出函数

TVM将需要从C++自动导出的函数打包成PackedFunc，然后通过宏TVM_REGISTER_GLOBAL注册到全局的一个map中。比如：
```c++
TVM_REGISTER_GLOBAL("_Var")
.set_body_typed([](std::string s, DataType t) {
    return VarNode::make(t, s);
  });
```

当Python加载编译好的动态库时，会自动查询map中静态注册的函数，每个函数都包装成Python中的Function对象，最终添加到Python模块中。Function重定义了函数调用接口，自动完成参数打包过程。
如果是在Python中动态注册的函数，则需要在Python中通过函数名和来查询PackedFunc，返回一个PackedFunc的handle（函数指针），并封装成Function。

```python
def get_global_func(name, allow_missing=False):
    handle = FunctionHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return Function(handle, False)

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)
```
注：TVMFuncGetGlobal是通过ctypes导出的C++接口，FunctionHandle是ctypes中表示void指针类型（c_void_p）。

### 从Python注册函数

由于TVM中PackedFunc的精心设计，我们只需要将Python中的函数转换成统一的函数原型void(const TVMArgs, TVMRetValue)，然后将函数转换成PackedFunc并动态地注册到全局的map中。

先将Python函数用ctypes转成int(TVMValue *, int *, int, void *, void *)的C函数。

```python
TVMPackedCFunc = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(TVMValue),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p)
```

然后通过TVMFuncCreateFromCFunc将上面的C函数转换成统一的PackedFunc函数。

```c++
int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
                           void* resource_handle,
                           TVMPackedCFuncFinalizer fin,
                           TVMFunctionHandle *out) {
  API_BEGIN();
  if (fin == nullptr) {
    *out = new PackedFunc(
        [func, resource_handle](TVMArgs args, TVMRetValue* rv) {
          int ret = func((TVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
                         args.num_args, rv, resource_handle);
          if (ret != 0) {
            throw dmlc::Error(TVMGetLastError() + ::dmlc::StackTrace());
          }
        });
  } else {
    ...
  }
  API_END();
}
```

最后通过接口TVMFuncRegisterGlobal注册到全局的map中。下面是从Python中注册一个函数，并在Python中调用的例子。

```python
targs = (10, 10.0, "hello")

@tvm.register_func
def my_packed_func(*args):
    assert(tuple(args) == targs)
    return 10

# Get it out from global function table
f = tvm.get_global_func("my_packed_func")
assert isinstance(f, tvm.nd.Function)
y = f(*targs)
assert y == 10
```
