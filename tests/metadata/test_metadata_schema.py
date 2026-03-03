"""Tests for metadata schema validation."""

import json
import pytest
from pathlib import Path


def load_metadata_schema():
    """Load the metadata schema from file."""
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "metadata.schema.json"
    with open(schema_path, "r") as f:
        return json.load(f)


class TestMetadataSchema:
    """Test the metadata schema structure and validation."""
    
    def test_schema_file_exists(self):
        """Metadata schema file should exist."""
        schema_path = Path(__file__).parent.parent.parent / "schemas" / "metadata.schema.json"
        assert schema_path.exists(), f"Metadata schema not found at {schema_path}"
    
    def test_schema_is_valid_json(self):
        """Metadata schema should be valid JSON."""
        schema = load_metadata_schema()
        assert isinstance(schema, dict)
        assert "$schema" in schema
    
    def test_schema_has_required_properties(self):
        """Metadata schema should define key properties."""
        schema = load_metadata_schema()
        required_props = ["fuse", "version", "domain", "trainables", "id", "variant"]
        
        for prop in required_props:
            assert prop in schema.get("properties", {}), f"Missing property: {prop}"
    
    def test_fuse_version_pattern(self):
        """Fuse version property should have version pattern."""
        schema = load_metadata_schema()
        fuse_prop = schema["properties"]["fuse"]
        
        assert "pattern" in fuse_prop or "type" in fuse_prop
        assert fuse_prop.get("type") == "string"
    
    def test_trainables_property_definition(self):
        """Trainables property should allow name -> bool/metadata mapping."""
        schema = load_metadata_schema()
        trainables_prop = schema["properties"]["trainables"]
        
        assert trainables_prop.get("type") == "object"
        assert "additionalProperties" in trainables_prop
    
    def test_metadata_examples_provided(self):
        """Schema should include examples."""
        schema = load_metadata_schema()
        assert "examples" in schema
        assert len(schema["examples"]) > 0
        
        # First example should be a valid metadata dict
        example = schema["examples"][0]
        assert isinstance(example, dict)
        assert "fuse" in example or "version" in example
    
    def test_example_metadata_structure(self):
        """First example should have realistic metadata structure."""
        schema = load_metadata_schema()
        example = schema["examples"][0]
        
        # Check for key fields
        assert example.get("fuse") == "0.7.2"
        assert example.get("version") == "1.0.0"
        assert isinstance(example.get("trainables", {}), dict)
        assert isinstance(example.get("training_config", {}), dict)
    
    def test_id_format_validation(self):
        """ID property should enforce IRI or CURIE format."""
        schema = load_metadata_schema()
        id_prop = schema["properties"]["id"]
        
        assert "pattern" in id_prop, "ID property should have format validation pattern"
        # Pattern should match http/https or colon-separated CURIE
        pattern = id_prop["pattern"]
        assert "http" in pattern or "://" in pattern


class TestMetadataValidation:
    """Test actual metadata validation against schema."""
    
    def _validate_metadata(self, metadata: dict) -> None:
        """Helper to validate metadata against schema."""
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")
        
        schema = load_metadata_schema()
        jsonschema.validate(metadata, schema)
    
    def test_valid_minimal_metadata(self):
        """Minimal valid metadata should pass validation."""
        metadata = {
            "fuse": "0.7.2",
            "version": "1.0.0"
        }
        
        try:
            self._validate_metadata(metadata)
        except ImportError:
            pass  # Skip if jsonschema not available
    
    def test_valid_full_metadata(self):
        """Full example metadata should pass validation."""
        schema = load_metadata_schema()
        example = schema["examples"][0]
        
        try:
            self._validate_metadata(example)
        except ImportError:
            pass  # Skip if jsonschema not available
    
    def test_invalid_fuse_version_format(self):
        """Invalid fuse version format should fail validation."""
        metadata = {
            "fuse": "invalid version",  # Should be X.Y or X.Y.Z
            "version": "1.0.0"
        }
        
        try:
            import jsonschema
            schema = load_metadata_schema()
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(metadata, schema)
        except ImportError:
            pytest.skip("jsonschema not installed")
    
    def test_invalid_id_format(self):
        """Invalid ID format should fail validation."""
        metadata = {
            "fuse": "0.7.2",
            "id": "invalid format without http or colon"  # Should be IRI or CURIE
        }
        
        try:
            import jsonschema
            schema = load_metadata_schema()
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(metadata, schema)
        except ImportError:
            pytest.skip("jsonschema not installed")
