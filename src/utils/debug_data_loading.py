#!/usr/bin/env python3
"""
Debug script to test NER data loading and identify tensor reshaping issues
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ner.data.processor import NERDataProcessor
from ner.data.loader import NERDataLoader

def debug_data_loading():
    """Debug the data loading process"""
    
    # Load configuration
    config_path = Path('data/ner/configs/countries/uae_google.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded: {config_path}")
    print(f"Labels config: {config['labels']}")
    
    # Initialize processor
    processor = NERDataProcessor()
    
    # Load training data
    train_file = Path(config['data']['train_file'])
    print(f"\nLoading training data from: {train_file}")
    
    if not train_file.exists():
        print(f"ERROR: Training file not found: {train_file}")
        return
    
    try:
        train_examples = processor.load_data_file(str(train_file))
        print(f"Loaded {len(train_examples)} training examples")
        
        # Check first few examples
        for i, example in enumerate(train_examples[:3]):
            print(f"\nExample {i}:")
            print(f"  Tokens: {len(example['tokens'])} - {example['tokens'][:5]}...")
            print(f"  Labels: {len(example['labels'])} - {example['labels'][:5]}...")
            
            if len(example['tokens']) != len(example['labels']):
                print(f"  ERROR: Token/label length mismatch!")
                return
    
    except Exception as e:
        print(f"ERROR loading training data: {e}")
        return
    
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
    print(f"\nLabel mappings ({len(label2id)} labels):")
    for label, idx in list(label2id.items())[:10]:
        print(f"  {label}: {idx}")
    
    # Test data loader
    print(f"\nTesting data loader...")
    try:
        data_loader = NERDataLoader(
            tokenizer_name=config['model']['pretrained_model'],
            label2id=label2id,
            max_length=config['data'].get('max_length', 512)
        )
        
        # Create dataset
        dataset = data_loader.create_dataset(train_examples[:10])  # Test with first 10 examples
        print(f"Created dataset with {len(dataset)} examples")
        
        # Test first example
        if len(dataset) > 0:
            first_item = dataset[0]
            print(f"\nFirst dataset item:")
            for key, value in first_item.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)} - {value if len(str(value)) < 100 else str(value)[:100]+'...'}")
        
        # Test dataloader
        dataloader = data_loader.create_dataloader(dataset, batch_size=2)
        print(f"\nTesting dataloader...")
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            for key, value in batch.items():
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
                
    except Exception as e:
        print(f"ERROR in data loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nData loading test completed successfully!")

if __name__ == '__main__':
    debug_data_loading()