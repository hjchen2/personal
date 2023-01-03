---
title: C++调用python
date: 2017-07-3 12:31:08
category: code
tags: [c++, python, embedding]
---


由于需要在组内新开发的一套机器学习框架上开发一个强化学习的demo，但目前开源的一些游戏环境都只提供了python接口，比如Gym。如果要使用Gym去做在线训练的话，就需要在C++代码中调用Python接口，因此找了些例子学习了一下如何使用Python C API。当然Python C API不是唯一的方式，也可以使用boost的Python模块，有时间再研究。

<!-- more -->

## hello python

```c++
#include <stdio.h>
#include <iostream>
#include "python/Python.h"

int main() {
    Py_Initialize();
    std::cout << "hello c++!" << std::endl;
    PyRun_SimpleString("print 'hello python!'");
    Py_Finalize();
    return 0;
}
```

编译：

```txt
g++ test.cpp -o test -lpython
```

执行：./test

```txt
hello c++!
hello python!
```



## 调用python脚本中的函数

```python
# test_add.py
def add(a, b):
    return a+b
```

```c++
#include <stdio.h>
#include <iostream>
#include "python/Python.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./exe integer1 integer2" << std::endl;
        return 1;
    }
    std::cerr << "hello c++!" << std::endl;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");

    PyRun_SimpleString("print 'hello python!'");
    PyObject* moduleName = PyString_FromString("test_add");
    PyObject* pModule = PyImport_Import(moduleName);
    if (!pModule) {
        std::cerr << "[ERROR] Python get module failed." << std::endl;
        return 1;
    }
    PyObject* pv = PyObject_GetAttrString(pModule, "add");
    if (!pv || !PyCallable_Check(pv)) {
        std::cerr << "[ERROR] Can't find function (add)" << std::endl;
        return 1;
    }

    PyObject* args = PyTuple_New(2);
    PyObject* arg1 = PyInt_FromLong(atoi(argv[1]));
    PyObject* arg2 = PyInt_FromLong(atoi(argv[2]));
    PyTuple_SetItem(args, 0, arg1);
    PyTuple_SetItem(args, 1, arg2);

    PyObject* pRet = PyObject_CallObject(pv, args);
    if (!pRet) {
        std::cerr << "[ERROR] Call funftion (add) failed" << std::endl;
        return 1;
    }
    long result = PyInt_AsLong(pRet);
    std::cout << "result: " << result << std::endl;

    Py_Finalize();
    return 0;
}
```

编译：

```txt
g++ test.cpp -o test -lpython
```

执行：./test 3 4

```txt
hello c++!
hello python!
result: 7
```



## Q学习的一个例子

```python
# tree.py
"""
author: Houjiang Chen
"""
import random

class q_learning:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.eps = 0.1
        self.alpha = 0.1
        self.q_table = [[0 for j in range(actions)] for i in range(states)]

    def get_action(self, current_state):
        max_action = self.q_table[current_state].index(max(self.q_table[current_state]))
        if random.uniform(0, 1) > self.eps:
            return max_action
        else:
            rest = [i for i in range(len(self.q_table[current_state])) if i != max_action]
            index = random.randint(0, len(rest) - 1)
            return rest[index]

    def update(self, current_state, action, next_state, reward, final):
        if not final:
            reward = reward + max(self.q_table[next_state])
        self.q_table[current_state][action] += self.alpha * (reward - self.q_table[current_state][action])


class environment:
    def __init__(self):
        self.level = 2
        self.actions = 2
        self.states = self.actions ** (self.level + 1) - 1
        self.final_states = self.actions ** self.level
        self.reward = {0 : [10, -10], 1 : [50, 100], 2 : [100, 150]}

    def next(self, current_state, action):
        """action: 0 or 1
           return: next_state reward, is_final
        """
        next = 2 * current_state + (action + 1)
        if next >= self.states - self.final_states:
            return None, self.reward[current_state][action], True
        else:
            return next, self.reward[current_state][action], False

    def reset(self):
        return random.randint(0, self.states - self.final_states - 1)


def main():
    env = environment()
    agent = q_learning(env.states, env.actions)

    episode = 0
    while episode < 10000:
        episode += 1
        print "episode: %d" % episode
        current_state = env.reset()
        while True:
            action = agent.get_action(current_state)
            next_state, reward, final = env.next(current_state, action)
            agent.update(current_state, action, next_state, reward, final)
            if final:
                break
            current_state = next_state

    print agent.q_table

if __name__ == '__main__':
    main()
```



```c++
#include <stdio.h>
#include <iostream>
#include "python2.7/Python.h"

PyObject* New_PyInstance(PyObject* cls, PyObject* args) {
    PyObject* pInstance = PyInstance_New(cls, args, NULL);
    if (!pInstance) {
        std::cerr << "new instance failed" << std::endl;
        exit(1);
    }
    return pInstance;
}

int main(int argc, char* argv[]) {
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");

    PyObject* moduleName = PyString_FromString("tree");
    PyObject* pModule = PyImport_Import(moduleName);
    if (!pModule) {
        std::cerr << "[ERROR] Python get module failed." << std::endl;
        return 1;
    }
    PyObject* pEnv = PyObject_GetAttrString(pModule, "environment");
    if (!pEnv) {
        std::cerr << "[ERROR] Can't find class (environment)" << std::endl;
        return 1;
    }

    PyObject* pEnvObject = New_PyInstance(pEnv, NULL);
    PyObject* pEnvLevel = PyObject_GetAttrString(pEnvObject, "level");
    if (!pEnvLevel) {
        std::cerr << "[ERROR] Env has no attr level" << std::endl;
        return 1;
    }
    PyObject* pEnvActions = PyObject_GetAttrString(pEnvObject, "actions");
    PyObject* pEnvStates = PyObject_GetAttrString(pEnvObject, "states");
    PyObject* pEnvFinalState = PyObject_GetAttrString(pEnvObject, "final_states");

    int level = PyInt_AsLong(pEnvLevel);
    int actions = PyInt_AsLong(pEnvActions);
    int states = PyInt_AsLong(pEnvStates);
    int final_state = PyInt_AsLong(pEnvFinalState);

    std::cout << "env level: " << level << std::endl;
    std::cout << "env actions: " << actions << std::endl;
    std::cout << "env states: " << states << std::endl;
    std::cout << "env final_state: " << final_state << std::endl;

    PyObject* pLearn = PyObject_GetAttrString(pModule, "q_learning");
    PyObject* pLearnArgs = Py_BuildValue("ii", states, actions);
    PyObject* pLearnObject = New_PyInstance(pLearn, pLearnArgs);
    PyObject* pLearnStates = PyObject_GetAttrString(pLearnObject, "states");
    PyObject* pLearnActions = PyObject_GetAttrString(pLearnObject, "actions");
    PyObject* pLearnEps = PyObject_GetAttrString(pLearnObject, "eps");

    int learn_states = PyInt_AsLong(pLearnStates);
    int learn_actions = PyInt_AsLong(pLearnActions);
    float learn_eps = PyFloat_AsDouble(pLearnEps);

    std::cout << "learn_states: " << learn_states << std::endl;
    std::cout << "learn_actions: " << learn_actions << std::endl;
    std::cout << "learn_eps: " << learn_eps << std::endl;

    PyObject* pEnvResetFunc = PyObject_GetAttrString(pEnvObject, "reset");
    PyObject* pEnvNextFunc = PyObject_GetAttrString(pEnvObject, "next");
    PyObject* pLearnGetActionFunc = PyObject_GetAttrString(pLearnObject, "get_action");
    PyObject* pLearnUpdateFunc = PyObject_GetAttrString(pLearnObject, "update");
    if (!pEnvNextFunc) {
        std::cerr << "[ERROR] env has no function named next" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    uint64_t episode = 0;
    for (episode = 0; episode < 10000; ++episode) {
        if (episode % 100 == 0)
            std::cout << "episode: " << episode << std::endl;
        PyObject* current_state = PyEval_CallObject(pEnvResetFunc, NULL);
        while (true) {
            PyObject* args1 = PyTuple_New(1);
            PyObject* args2 = PyTuple_New(2);
            PyTuple_SetItem(args1, 0, current_state);
            PyObject* action = PyEval_CallObject(pLearnGetActionFunc, args1);
            PyTuple_SetItem(args2, 0, current_state);
            PyTuple_SetItem(args2, 1, action);
            PyObject* ret = PyEval_CallObject(pEnvNextFunc, args2);
            PyObject* next_state = PyTuple_GetItem(ret, 0);
            PyObject* final = PyTuple_GetItem(ret ,2);
            PyObject* args3 = PyTuple_New(5);
            PyTuple_SetItem(args3, 0, current_state);
            PyTuple_SetItem(args3, 1, action);
            PyTuple_SetItem(args3, 2, next_state);
            PyTuple_SetItem(args3, 3, PyTuple_GetItem(ret, 1));
            PyTuple_SetItem(args3, 4, final);

            PyEval_CallObject(pLearnUpdateFunc, args3);
            if (PyObject_IsTrue(final)) {
                break;
            }
            current_state = next_state;
            if (args3)
                Py_DECREF(args3);
        }
    }
    PyObject* pLearnQTable = PyObject_GetAttrString(pLearnObject, "q_table");
    for (int i = 0; i < PyList_Size(pLearnQTable); ++i) {
        std::cout << "state " << i << std::endl;
        PyObject* term = PyList_GetItem(pLearnQTable, i);
        if (PyList_Check(term)) {
            for (int j = 0; j < PyList_Size(term); ++j) {
                std::cout << "    direct: " << j << ", " << "Qvalue: "
                          << PyFloat_AsDouble(PyList_GetItem(term, j)) << std::endl;
            }
        }
    }
    Py_Finalize();
    return 0;
}
```

编译：

```txt
g++ test.cpp -o test -I../python2.7.12/include -L../python2.7.12/lib -lpython2.7
```

执行：./test

```txt
env level: 2
env actions: 2
env states: 7
env final_state: 4
learn_states: 7
learn_actions: 2
learn_eps: 0.1

episode: 0
episode: 100
episode: 200
episode: 300
episode: 400
episode: 500
episode: 600
episode: 700
episode: 800
episode: 900
episode: 1000
episode: 1100
episode: 1200
episode: 1300
episode: 1400
episode: 1500
episode: 1600
episode: 1700
episode: 1800
episode: 1900
episode: 2000
episode: 2100
episode: 2200
episode: 2300
episode: 2400
episode: 2500
episode: 2600
episode: 2700
episode: 2800
episode: 2900
episode: 3000
episode: 3100
episode: 3200
episode: 3300
episode: 3400
episode: 3500
episode: 3600
episode: 3700
episode: 3800
episode: 3900
episode: 4000
episode: 4100
episode: 4200
episode: 4300
episode: 4400
episode: 4500
episode: 4600
episode: 4700
episode: 4800
episode: 4900
episode: 5000
episode: 5100
episode: 5200
episode: 5300
episode: 5400
episode: 5500
episode: 5600
episode: 5700
episode: 5800
episode: 5900
episode: 6000
episode: 6100
episode: 6200
episode: 6300
episode: 6400
episode: 6500
episode: 6600
episode: 6700
episode: 6800
episode: 6900
episode: 7000
episode: 7100
episode: 7200
episode: 7300
episode: 7400
episode: 7500
episode: 7600
episode: 7700
episode: 7800
episode: 7900
episode: 8000
episode: 8100
episode: 8200
episode: 8300
episode: 8400
episode: 8500
episode: 8600
episode: 8700
episode: 8800
episode: 8900
episode: 9000
episode: 9100
episode: 9200
episode: 9300
episode: 9400
episode: 9500
episode: 9600
episode: 9700
episode: 9800
episode: 9900
state 0
    direct: 0, Qvalue: 110
    direct: 1, Qvalue: 140
state 1
    direct: 0, Qvalue: 50
    direct: 1, Qvalue: 100
state 2
    direct: 0, Qvalue: 100
    direct: 1, Qvalue: 150
state 3
    direct: 0, Qvalue: 0
    direct: 1, Qvalue: 0
state 4
    direct: 0, Qvalue: 0
    direct: 1, Qvalue: 0
state 5
    direct: 0, Qvalue: 0
    direct: 1, Qvalue: 0
state 6
    direct: 0, Qvalue: 0
    direct: 1, Qvalue: 0
```



## 参考资料

Python/C API Reference Manual:  https://docs.python.org/2/c-api/index.html
