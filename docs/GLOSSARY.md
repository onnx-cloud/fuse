# Fuse Project Glossary

This glossary defines key terms and concepts used throughout the Fuse ONNX DSL project and documentation.

**Cognitive Architecture**
: A modular, reusable structure for building machine learning models and agents using composable operators and graphs.

**Operator**
: A fusible subgraph or composition pattern for ML models, agents, or fleets. Operators are weight-less models that can be composed to build larger architectures.


**Graph**
: A named, reusable computation consisting of nodes (operators), parameters, and connections. Graphs can represent models, submodels, or computation blocks.

**Node**
: An instance of an operator or function within a graph. Nodes define computation and data flow.

**Pragma**
: A special annotation (e.g., `@note`, `@domain`, `@meta`) used in Fuse source files to specify metadata, documentation, or configuration.

**@domain**
: Pragma specifying the namespace or domain for a model or operator, ensuring unique naming and modularity.

**@note**
: Pragma for attaching documentation strings to models, operators, or parameters.

**@meta**
: Pragma for attaching arbitrary metadata to a model or operator, serialized into ONNX `metadata_props`.

**@train / @frozen**
: Pragmas for marking weights as trainable or frozen (non-trainable) in the model.

**@type**
: Pragma specifying the semantic type or schema URI for a model, operator, or parameter.

**@id**
: Pragma specifying a unique identifier (URI) for a model, operator, or graph.

**@attribution**
: Pragma for citing sources, inspiration, or authorship for a model or operator.

**@version**
: Pragma specifying the version of a model, operator, or the Fuse DSL itself.

**@author**
: Pragma specifying the IRI of author(s) of a model or operator.

**@license**
: Pragma specifying the IRI of license for a model or operator.

**Fuse File**
: A source file written in the Fuse DSL, typically with a `.fuse` extension, describing models, operators, and graphs.

**Lowering**
: The process of converting Fuse source code into ONNX IR and ModelProto, including type/shape checking, constant folding, and deterministic naming.

**Sealing**
: Embedding a deterministic hash (seal) into the ONNX model metadata to ensure model integrity and reproducibility.

**External Data**
: Large tensor initializers stored outside the ONNX file, referenced via `external_data` fields for efficient model storage.

**TTL (RDF/Turtle)**
: A format for representing ONNX models and metadata as RDF triples, supporting semantic web and knowledge graph use cases.

**Golden Test**
: A test that checks the output of the compiler or lowering pipeline against a known-good (golden) ONNX model or artifact.

**Zoo**
: A local or remote repository of reusable Fuse models, operators, and graphs, supporting publishing and discovery.

**Namespace**
: A unique domain or prefix used to avoid naming collisions between models, operators, and graphs.

**SSA-style Naming**
: Static single assignment naming convention used for deterministic and reproducible model outputs.

**Trainable**
: A parameter or weight that is updated during training, as opposed to frozen (non-trainable) parameters.

**Fusion**
: The process of importing and merging external ONNX models or variants into a composite graph, with deterministic wiring.

**Config File**
: A JSON file specifying CLI and model configuration, validated against `schemas/fuse.config.schema.json`.

