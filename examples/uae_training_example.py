#!/usr/bin/env python3
"""UAE DAPT Training Example

This script demonstrates how to use the DAPT CLI tool to train
a model specifically for UAE Arabic dialect and domain.

Usage:
    python examples/uae_training_example.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dapt.config.manager import ConfigManager
from dapt.data.processor import DataProcessor
from dapt.engine.trainer import DAPTTrainingEngine
from dapt.evaluation.manager import EvaluationManager
from dapt.models.manager import ModelManager

def setup_uae_environment():
    """Setup environment for UAE training"""
    print("🇦🇪 Setting up UAE DAPT Training Environment...")
    
    # Create necessary directories
    directories = [
        "outputs/uae",
        "models/uae", 
        "logs/uae",
        "cache/uae",
        "data/uae"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("✅ Environment setup complete!")

def validate_uae_data():
    """Validate UAE training data"""
    print("\n📊 Validating UAE Training Data...")
    
    # Check if data files exist
    data_files = [
        "data/uae/train.jsonl",
        "data/uae/validation.jsonl", 
        "data/uae/test.jsonl"
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            # Count lines in file
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"✓ {data_file}: {lines} samples")
        else:
            print(f"❌ Missing: {data_file}")
            return False
    
    print("✅ Data validation complete!")
    return True

def run_uae_training_cli():
    """Run UAE training using CLI commands"""
    print("\n🚀 Starting UAE Model Training via CLI...")
    
    # CLI commands to execute
    commands = [
        # 1. Validate configuration
        ["python", "dapt_cli.py", "config", "validate", "--country", "uae"],
        
        # 2. Process and validate data
        ["python", "dapt_cli.py", "data", "process", "--country", "uae", "--validate"],
        
        # 3. Start training
        ["python", "dapt_cli.py", "train", "--country", "uae", "--epochs", "3", "--verbose"],      
        
        # 4. Evaluate model
        ["python", "dapt_cli.py", "evaluate", "--country", "uae", "--dataset", "test"],
        
        # 5. Check training status
        ["python", "dapt_cli.py", "status", "--country", "uae"]
    ]
    
    for i, command in enumerate(commands, 1):
        print(f"\n📋 Step {i}: {' '.join(command)}")
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                print(f"✅ Step {i} completed successfully")
                if result.stdout:
                    print(f"Output: {result.stdout[:200]}..." if len(result.stdout) > 200 else f"Output: {result.stdout}")
            else:
                print(f"❌ Step {i} failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Step {i} failed with exception: {str(e)}")
            return False
    
    print("\n✅ UAE training pipeline completed successfully!")
    return True

def run_uae_training_programmatic():
    """Run UAE training using programmatic API"""
    print("\n🔧 Starting UAE Model Training via API...")
    
    try:
        # 1. Load configuration
        print("📋 Loading UAE configuration...")
        config_manager = ConfigManager()
        config = config_manager.load_config("uae")
        print("✅ Configuration loaded")
        
        # 2. Process data
        print("\n📊 Processing training data...")
        data_processor = DataProcessor(config, config_manager.global_config)
        
        # Load and validate data
        train_data = data_processor.load_data("data/uae/train.jsonl")
        val_data = data_processor.load_data("data/uae/validation.jsonl")
        
        print(f"✓ Loaded {len(train_data)} training samples")
        print(f"✓ Loaded {len(val_data)} validation samples")
        
        # Validate data
        validation_results = data_processor.validate_data(train_data)
        if not validation_results['is_valid']:
            print(f"❌ Data validation failed: {validation_results['errors']}")
            return False
        
        print("✅ Data processing complete")
        
        # 3. Initialize training engine
        print("\n🚂 Initializing training engine...")
        trainer = DAPTTrainingEngine(config, config_manager.global_config)
        
        # Prepare datasets
        train_dataset = data_processor.prepare_dataset(train_data)
        val_dataset = data_processor.prepare_dataset(val_data)
        
        print("✅ Training engine initialized")
        
        # 4. Start training (reduced epochs for demo)
        print("\n🏋️ Starting model training...")
        config['training']['num_epochs'] = 2  # Reduce for demo
        
        training_results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        if training_results['status'] == 'completed':
            print("✅ Training completed successfully")
            print(f"📊 Final metrics: {training_results.get('final_metrics', {})}")
        else:
            print(f"❌ Training failed: {training_results.get('error', 'Unknown error')}")
            return False
        
        # 5. Evaluate model
        print("\n📈 Evaluating trained model...")
        evaluator = EvaluationManager(config, config_manager.global_config)
        
        test_data = data_processor.load_data("data/uae/test.jsonl")
        test_dataset = data_processor.prepare_dataset(test_data)
        
        eval_results = evaluator.evaluate_model(
            model_path=training_results['model_path'],
            test_dataset=test_dataset,
            dataset_name="uae_test"
        )
        
        print("✅ Evaluation completed")
        print(f"📊 Evaluation metrics: {eval_results.get('metrics', {})}")
        
        # 6. Save model
        print("\n💾 Saving trained model...")
        model_manager = ModelManager(config, config_manager.global_config)
        
        saved_model_info = model_manager.save_model(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            model_name=f"uae_bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metrics=eval_results.get('metrics', {})
        )
        
        print(f"✅ Model saved: {saved_model_info['model_path']}")
        
        print("\n🎉 UAE training pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_uae_report():
    """Generate training and evaluation report"""
    print("\n📄 Generating UAE Training Report...")
    
    try:
        # Use CLI to generate report
        command = [
            "python", "dapt_cli.py", "evaluate", 
            "--country", "uae", 
            "--generate-report",
            "--report-format", "html", "pdf"
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print("✅ Report generated successfully")
            print(f"📄 Report location: {result.stdout}")
        else:
            print(f"❌ Report generation failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Report generation failed: {str(e)}")

def main():
    """Main execution function"""
    print("🇦🇪 UAE DAPT Training Example")
    print("=" * 50)
    
    # Setup environment
    setup_uae_environment()
    
    # Validate data
    if not validate_uae_data():
        print("❌ Data validation failed. Please check data files.")
        return
    
    # Choose training method
    print("\n🤔 Choose training method:")
    print("1. CLI-based training (recommended)")
    print("2. Programmatic API training")
    print("3. Generate report only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        success = run_uae_training_cli()
    elif choice == "2":
        success = run_uae_training_programmatic()
    elif choice == "3":
        generate_uae_report()
        return
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    if success:
        # Generate report
        generate_uae_report()
        
        print("\n🎉 UAE DAPT Training Example Completed!")
        print("\n📋 Next Steps:")
        print("1. Check the generated model in models/uae/")
        print("2. Review training logs in logs/uae/")
        print("3. Examine evaluation reports in outputs/uae/")
        print("4. Use the trained model for inference")
        
        # Show model usage example
        print("\n💡 Model Usage Example:")
        print("```python")
        print("from dapt.models.loader import ModelLoader")
        print("")
        print("# Load trained UAE model")
        print("loader = ModelLoader(config, global_config)")
        print("model, tokenizer = loader.load_model('path/to/uae/model')")
        print("")
        print("# Use for inference")
        print("text = 'محمد أحمد يعمل في دبي'")
        print("predictions = model.predict(text)")
        print("```")
    else:
        print("\n❌ Training failed. Please check the logs for details.")

if __name__ == "__main__":
    main()