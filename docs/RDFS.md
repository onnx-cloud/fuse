# RDF/Turtle Export for Fuse 🔧📄

## Core vocabulary

- Classes
  - `onnx:Model`  (a whole ModelProto)
  - `onnx:Graph`  (a graph in the model)
  - `onnx:Node`   (an operator node)
  - `onnx:Initializer`
  - `onnx:TensorType` (dtype info)
  - `onnx:TensorShape` (shape info)

- Properties
  - `onnx:hasGraph` (Model → Graph)
  - `onnx:hasNode`  (Graph → Node)
  - `onnx:domain`  (fn → domain IRI or literal)
  - `onnx:opType`  (fn → operator name, e.g., "Conv")
  - `onnx:inputs` (fn → literal list of input names, compact summary)
  - `onnx:outputs` (fn → literal list of output names, compact summary)
  - `onnx:name`    (fn or Initializer → literal name)
  - `onnx:hasInitializer` (Graph → Initializer)
  - `onnx:dtype` / `onnx:shape` (tensor metadata)
  - `onnx:opset` (Model metadata)
  - `onnx:trainable` if needed (serialized as `xsd:boolean`)

Notes:
- The previous `onnx:OperatorInput` / `onnx:OperatorOutput` resources are intentionally no longer emitted by default. TTL now exports the **fusable surface** (a compact node-level graph summary mapping nodes to their input/output names) by default to make TTL outputs concise and easy to consume. If fine-grained operator input/output resources are required, the exporter can be extended to support it as an opt-in behavior (not enabled by default).

- Special metadata semantics:
  - `@type` when provided in Fuse source is emitted as an RDF `rdf:type` triple on the `:model/<id>` resource. **It should be either an absolute IRI** (starts with `http://` or `https://`) **or a CURIE** (`prefix:local`). CURIEs are accepted and emitted as-is; if you use a CURIE, declare the matching prefix via the TTL `user_ns`/`user_ns_uri` parameters or use a standard prefix (e.g., `rdf:`). Non-IRI and non-CURIE values will be rejected by the TTL exporter. In addition, the TTL exporter supports a `strict=True` mode: when enabled, unknown CURIE prefixes will raise an error rather than a warning.
  - `@id` is emitted as a `skos:exactMatch` linking the model resource to the author's identifier. **It should be either an absolute IRI or a CURIE**; same rules and warnings apply. Note: `@id` is not used as the canonical model URI — the model URI remains deterministic (model hash or explicit user namespace).

## Mapping rules (deterministic)

1. **Model & Graph URIs**
   - `:model/<model_file_hash>` or `:model/<stable_name>` → `a onnx:Model ; onnx:opset "13"^^xsd:integer ; onnx:hasGraph :graph/0 ; ... .`
   - Use a stable hash (sha256 of the normalized ModelProto bytes) or deterministic id from GraphContext.

2. **Graph → fn order**
   - Emit `onnx:hasNode` triples in a deterministic order: nodes ordered by GraphContext stable SSA-style names and then by index.

3. **fn URIs**
   - Use `:node/<stable-name>` (stable name allocator from GraphContext), e.g. `my:GPT a onnx:fn ; onnx:domain my:ai.onnx ; ... .`
   - Include `onnx:opType` and `onnx:domain` as literals or IRIs.

4. **Inputs / Outputs (compact node-level form)**
   - By default we emit a compact node-level summary rather than separate per-operator input/output resources to keep TTL concise and easy to consume:
     - `:node/<name> onnx:inputs "[x,y,...]" ; onnx:inputCount 2 ; onnx:outputs "[z]" ; onnx:outputCount 1 ;`
   - If full, per-input resources are required, export can be configured to emit `onnx:OperatorInput` / `onnx:OperatorOutput` resources deterministically (same deterministic labeling rules apply).

5. **Attributes & Initializers**
   - Emit each initializer as `:init/<stable-name> a onnx:Initializer ; onnx:name "W" ; onnx:shape "[1,3,3]" ; onnx:dtype "float32" .`
   - Initializers represent internal constants and are not exposed as `onnx:GraphInput` by default. Only initializers explicitly marked as trainable are emitted as `:init/` resources in TTL.
   - If external data present, include `onnx:external "path" ; onnx:external_offset 0^^xsd:integer ;`.

6. **Types and shapes**
   - Use `xsd:integer` for dims when known; if unknown, omit dim or use `onnx:unknownDim "true"^^xsd:boolean`.

7. **Literals**
   - Booleans: always `"true"^^xsd:boolean` / `"false"^^xsd:boolean`.
   - Strings: plain literals unless they contain characters requiring `xsd:string`. We default to declaring things over strings.

## Deterministic serialization🔒

- Always **sort prefixes** in a canonical order (e.g., alphabetical), and emit a stable prefix section at the top.
- **Sort triples** lexicographically by (subject IRI or blank label, predicate IRI, object lexical form). This avoids RDF-normalization complexity but achieves deterministic bytes for stable graphs.
- Avoid random blank-fn labels. If blank nodes are used, generate labels from stable components (model hash, fn index, role).
- Represent complex structures (lists, tensors) using named resources rather than RDF list constructs when determinism is required.

