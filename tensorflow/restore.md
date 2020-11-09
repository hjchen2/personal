```python
from __future__ import division
import sys
import numpy as np
import tensorflow as tf

model_path = "../../../Downloads/encode_large.pb"

graph = tf.Graph()
with graph.as_default():
  graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)

inputs = []
outputs_set = set(graph.get_operations())
for op in graph.get_operations():
  if len(op.inputs) == 0 and op.type != "Const":
    inputs.append(op)
  else:
    for tensor in op.inputs:
      if tensor.op in outputs_set:
        outputs_set.remove(tensor.op)

for input in inputs:
    print("Input: %s" % input.name)

outputs = list(outputs_set)
for output in outputs:
    print("Output: %s" % output.name)

output_name = "seq2seq/encoder_1/layer_0/multi_head/LayerNorm/batchnorm/add_1"

with tf.Session(graph=graph) as sess:
    tensor = graph.get_tensor_by_name("%s:0" % inputs[0].name)
    output = graph.get_tensor_by_name("import/%s:0" % output_name)
    array = np.ones([1, 250, 560], dtype=np.float32)
    output = sess.run(output, feed_dict={tensor: array})
    print output.shape
    print np.sum(output)
```
