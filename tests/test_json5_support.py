#!/usr/bin/env python3
"""Test script to verify JSON5 support in ConfigManager"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ner.config.manager import ConfigManager

def test_json5_support():
    """Test JSON5 support in ConfigManager"""
    print("Testing JSON5 support in ConfigManager...")
    
    # Create a test JSON5 template
    test_template_content = '''{
    // This is a JSON5 template with comments
    "country": {
        "code": "test",
        "name": "TEST"
    },
    "model": {
        "name": "test-model",
        "type": "bert",
        "pretrained_model": "bert-base-multilingual-cased"
    },
    "training": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5
    },
    "data": {
        "train_file": "train.json",
        "val_file": "val.json"
    },
    "labels": {
        "num_labels": 3,
        "label_names": ["O", "B-PER", "I-PER"],
        "label_mapping": {"O": 0, "B-PER": 1, "I-PER": 2}
    },
    "evaluation": {
        "metrics": ["precision", "recall", "f1"]
    },
    "output": {
        "model_dir": "models/test"
    },
    "hardware": {
        "device": "auto"
    },
    "logging": {
        "level": "INFO"
    }
}'''
    
    # Create test directories
    test_dir = Path("test_config")
    templates_dir = test_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSON5 template
    json5_template_path = templates_dir / "test.json5"
    with open(json5_template_path, 'w', encoding='utf-8') as f:
        f.write(test_template_content)
    
    try:
        # Test ConfigManager with JSON5 support
        config_manager = ConfigManager(str(test_dir))
        
        # Test 1: List templates (should include .json5 files)
        print("\n1. Testing list_templates()...")
        templates = config_manager.list_templates()
        print(f"Available templates: {templates}")
        assert "test" in templates, "JSON5 template not found in list"
        print("✓ JSON5 template found in list")
        
        # Test 2: Load JSON5 template
        print("\n2. Testing load_template_config()...")
        config = config_manager.load_template_config("test")
        print(f"Loaded config keys: {list(config.keys())}")
        assert "country" in config, "Country section not found"
        assert "model" in config, "Model section not found"
        print("✓ JSON5 template loaded successfully")
        
        # Test 3: Create country config from JSON5 template
        print("\n3. Testing create_country_config()...")
        country_config = config_manager.create_country_config("test_country", "test")
        assert country_config["country"]["code"] == "test_country"
        assert country_config["country"]["name"] == "TEST_COUNTRY"
        print("✓ Country config created from JSON5 template")
        
        # Test 4: Load external template (JSON format)
        print("\n4. Testing load_external_template()...")
        external_template_path = test_dir / "external_template.json"
        # Create valid JSON (without comments)
        import json as json_module
        json_data = {
            "country": {
                "code": "test",
                "name": "TEST"
            },
            "model": {
                "name": "test-model",
                "type": "bert",
                "pretrained_model": "bert-base-multilingual-cased"
            },
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 2e-5
            },
            "data": {
                "train_file": "train.json",
                "val_file": "val.json"
            },
            "labels": {
                "num_labels": 3,
                "label_names": ["O", "B-PER", "I-PER"],
                "label_mapping": {"O": 0, "B-PER": 1, "I-PER": 2}
            },
            "evaluation": {
                "metrics": ["precision", "recall", "f1"]
            },
            "output": {
                "model_dir": "models/test"
            },
            "hardware": {
                "device": "auto"
            },
            "logging": {
                "level": "INFO"
            }
        }
        json_content = json_module.dumps(json_data, indent=2)
        with open(external_template_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        external_config = config_manager.load_external_template(str(external_template_path))
        assert "country" in external_config, "External template not loaded correctly"
        print("✓ External JSON template loaded successfully")
        
        # Test 5: Create config with external template
        print("\n5. Testing create_country_config() with external template...")
        external_country_config = config_manager.create_country_config(
            "external_test", 
            external_template_path=str(external_template_path)
        )
        assert external_country_config["country"]["code"] == "external_test"
        print("✓ Country config created from external template")
        
        print("\n🎉 All tests passed! JSON5 support is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    success = test_json5_support()
    sys.exit(0 if success else 1)