# ONNX Metadata Synonyms Mapping

The following table shows the supported ONNX metadata fields, their Fuse internal names, and the corresponding pragma or annotation. Synonyms for ONNX keys are supported.

| ONNX name (synonyms supported) | Fuse internal name | @pragma |
|---|---:|---|
| doc_string | doc | @note |
| domain | namespace | @domain |
| metadata_props | metadata | @meta |
| model_author | author | @author |
| model_license | license | @license |
| model_version | version | @version |
| producer_name | producer | 'onnx-fuse' |
| producer_version | producer_version | '1.2' |

> Synonyms: The parser and exporter accept both ONNX and Fuse names for metadata fields wherever possible.
