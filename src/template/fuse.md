---
title: {{graph.metadata.title}}
description: {{graph.metadata.description}}
---
# Operator: {{graph.name}}

{{graph.metadata.description}}

## {{graph.metadata.title || graph.name }}

{{graph.metadata.notes}}

### See also
{{if flag.dot}}
(dot){[{{file.name}}.dot] - 
{{/if}}


### Operator Architecture

```mermaid
{{ast.graph}}
```

### Operator Source

```fuse
{{fuse.code}}
```

### Operator Metrics

{{fuse.metrics}}