#!/usr/bin/env python3
"""
Debug script to test model initialization and identify the num_labels issue
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ner.models.model import BertNERModel

def debug_model_init():
    """Debug the model initialization process"""
    
    # Load configuration
    config_path = Path('data/ner/configs/countries/uae_google.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded: {config_path}")
    print(f"Labels config: {config['labels']}")
    
    # Create label mappings
    labels_config = config['labels']
    if 'entities' in labels_config:
        entities = labels_config['entities']
        bio_labels = ['O'] + [f'B-{entity}' for entity in entities] + [f'I-{entity}' for entity in entities]
    elif 'label_names' in labels_config:
        label_names = labels_config['label_names']
        entities = set()
        for label in label_names:
            if label.startswith('B-') or label.startswith('I-'):
                entity = label[2:]
                entities.add(entity)
        entities = sorted(list(entities))
        bio_labels = label_names
    else:
        print("ERROR: No entities or label_names found in config")
        return
    
    label2id = {label: idx for idx, label in enumerate(bio_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(bio_labels)
    
    print(f"\nLabel mappings ({num_labels} labels):")
    for label, idx in list(label2id.items())[:10]:
        print(f"  {label}: {idx}")
    
    print(f"\nnum_labels: {num_labels}")
    
    # Test model initialization
    print(f"\nTesting model initialization...")
    try:
        model = BertNERModel.from_pretrained(
            pretrained_model_name_or_path=config['model']['pretrained_model'],
            num_labels=num_labels,
            dropout=config['model'].get('dropout', 0.1)
        )
        
        print(f"Model created successfully!")
        print(f"Model num_labels: {model.num_labels}")
        print(f"Model config num_labels: {model.config.num_labels}")
        print(f"Classifier layer: {model.classifier}")
        print(f"Classifier weight shape: {model.classifier.weight.shape}")
        print(f"Classifier bias shape: {model.classifier.bias.shape}")
        
    except Exception as e:
        print(f"ERROR in model initialization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nModel initialization test completed successfully!")

if __name__ == '__main__':
    debug_model_init()