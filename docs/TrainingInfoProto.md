TrainingInfoProto

Training information TrainingInfoProto stores information for training a model. In particular, this defines two functionalities: an initialization-step and a training-algorithm-step. Initialization resets the model back to its original state as if no training has been performed. Training algorithm improves the model based on input data.

The semantics of the initialization-step is that the initializers in ModelProto.graph and in TrainingInfoProto.algorithm are first initialized as specified by the initializers in the graph, and then updated by the "initialization_binding" in every instance in ModelProto.training_info.

The field "algorithm" defines a computation graph which represents a training algorithm's step. After the execution of a TrainingInfoProto.algorithm, the initializers specified by "update_binding" may be immediately updated. If the targeted training algorithm contains consecutive update steps (such as block coordinate descent methods), the user needs to create a TrainingInfoProto for each step.

Value parameters
algorithm
This field represents a training algorithm step. Given required inputs, it computes outputs to update initializers in its own or inference graph's initializer lists. In general, this field contains loss node, gradient node, optimizer node, increment of iteration count. An execution of the training algorithm step is performed by executing the graph obtained by combining the inference graph (namely "ModelProto.graph") and the "algorithm" graph. That is, the actual the actual input/initializer/output/node/value_info/sparse_initializer list of the training graph is the concatenation of "ModelProto.graph.input/initializer/output/node/value_info/sparse_initializer" and "algorithm.input/initializer/output/node/value_info/sparse_initializer" in that order. This combined graph must satisfy the normal ONNX conditions. 

Now, let's provide a visualization of graph combination for clarity. Let the inference graph (i.e., "ModelProto.graph") be tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d and the "algorithm" graph be tensor_d -> Add -> tensor_e The combination process results tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d -> Add -> tensor_e Notice that an input of a fn in the "algorithm" graph may reference the output of a fn in the inference graph (but not the other way round). Also, inference fn cannot reference inputs of "algorithm". With these restrictions, inference graph can always be run independently without training information. By default, this field is an empty graph and its evaluation does not produce any output. Evaluating the default training step never update any initializers.

initialization
This field describes a graph to compute the initial tensors upon starting the training process. Initialization graph has no input and can have multiple outputs. Usually, trainable tensors in neural networks are randomly initialized. To achieve that, for each tensor, the user can put a random number operator such as RandomNormal or RandomUniform in TrainingInfoProto.initialization.fn and assign its random output to the specific tensor using "initialization_binding". This graph can also set the initializers in "algorithm" in the same TrainingInfoProto; a use case is resetting the number of training iteration to zero. By default, this field is an empty graph and its evaluation does not produce any output. Thus, no initializer would be changed by default.

initializationBinding
This field specifies the bindings from the outputs of "initialization" to some initializers in "ModelProto.graph.initializer" and the "algorithm.initializer" in the same TrainingInfoProto. See "update_binding" below for details. By default, this field is empty and no initializer would be changed by the execution of "initialization".

updateBinding
Gradient-based training is usually an iterative procedure. In one gradient descent iteration, we apply x = x - r * g where "x" is the optimized tensor, "r" stands for learning rate, and "g" is gradient of "x" with respect to a chosen loss. 

To avoid adding assignments into the training graph, we split the update equation into y = x - r * g x = y The user needs to save "y = x - r * g" into TrainingInfoProto.algorithm. To tell that "y" should be assigned to "x", the field "update_binding" may contain a key-value pair of strings, "x" (key of StringStringEntryProto) and "y" (value of StringStringEntryProto). For a neural network with multiple trainable (mutable) tensors, there can be multiple key-value pairs in "update_binding". The initializers appears as keys in "update_binding" are considered mutable variables. This implies some behaviors as described below.

We have only unique keys in all "update_binding"s so that two variables may not have the same name. This ensures that one variable is assigned up to once.
The keys must appear in names of "ModelProto.graph.initializer" or "TrainingInfoProto.algorithm.initializer".
The values must be output names of "algorithm" or "ModelProto.graph.output".
Mutable variables are initialized to the value specified by the corresponding initializer, and then potentially updated by "initializer_binding"s and "update_binding"s in "TrainingInfoProto"s. This field usually contains names of trainable tensors (in ModelProto.graph), optimizer states such as momentums in advanced stochastic gradient methods (in TrainingInfoProto.graph), and number of training iterations (in TrainingInfoProto.graph). By default, this field is empty and no initializer would be changed by the execution of "algorithm".
Attributes
Companion
object
Graph
Show graph
Supertypes
trait Updatable[TrainingInfoProto]
trait GeneratedMessage
trait Serializable
trait Product
trait Equals
