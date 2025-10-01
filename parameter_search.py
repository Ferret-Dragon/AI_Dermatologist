"""
Parameter search script for model optimization.
Tests different hyperparameter combinations and records results.
"""

import os
import json
from datetime import datetime
import subprocess
import sys

# Define parameter configurations to test
CONFIGS = [
    {
        "name": "baseline",
        "IMAGE_SIZE": 80,
        "CONV1_CHANNELS": 8,
        "CONV2_CHANNELS": 16,
        "FC1_SIZE": 64,
        "DROPOUT_RATE": 0.5,
        "NUM_EPOCHS": 5,
        "LEARNING_RATE": 5e-4,
        "BATCH_SIZE": 512,
    },
    {
        "name": "smaller_model",
        "IMAGE_SIZE": 60,
        "CONV1_CHANNELS": 6,
        "CONV2_CHANNELS": 12,
        "FC1_SIZE": 48,
        "DROPOUT_RATE": 0.5,
        "NUM_EPOCHS": 5,
        "LEARNING_RATE": 5e-4,
        "BATCH_SIZE": 512,
    },
    {
        "name": "tiny_model",
        "IMAGE_SIZE": 50,
        "CONV1_CHANNELS": 4,
        "CONV2_CHANNELS": 8,
        "FC1_SIZE": 32,
        "DROPOUT_RATE": 0.5,
        "NUM_EPOCHS": 5,
        "LEARNING_RATE": 5e-4,
        "BATCH_SIZE": 512,
    },
    {
        "name": "more_epochs",
        "IMAGE_SIZE": 80,
        "CONV1_CHANNELS": 8,
        "CONV2_CHANNELS": 16,
        "FC1_SIZE": 64,
        "DROPOUT_RATE": 0.5,
        "NUM_EPOCHS": 10,
        "LEARNING_RATE": 5e-4,
        "BATCH_SIZE": 512,
    },
    {
        "name": "higher_lr",
        "IMAGE_SIZE": 80,
        "CONV1_CHANNELS": 8,
        "CONV2_CHANNELS": 16,
        "FC1_SIZE": 64,
        "DROPOUT_RATE": 0.5,
        "NUM_EPOCHS": 5,
        "LEARNING_RATE": 1e-3,
        "BATCH_SIZE": 512,
    },
]


def create_modified_starter_code(config, output_file="temp_starterCode.py"):
    """
    Create a modified version of starterCode.py with the given parameters.
    """
    with open("starterCode.py", "r") as f:
        lines = f.readlines()

    # Find the hyperparameters section and replace values
    modified_lines = []
    in_hyperparams = False

    for line in lines:
        if "# HYPERPARAMETERS & CONFIG" in line:
            in_hyperparams = True
        elif in_hyperparams and line.strip().startswith("# Paths and URL"):
            in_hyperparams = False

        # Replace parameter values
        if in_hyperparams:
            for param_name, param_value in config.items():
                if param_name == "name":
                    continue
                if line.strip().startswith(f"{param_name} ="):
                    line = f"{param_name} = {param_value}\n"
                    break

        modified_lines.append(line)

    # Write modified file
    with open(output_file, "w") as f:
        f.writelines(modified_lines)

    return output_file


def run_experiment(config):
    """
    Run an experiment with the given configuration.
    Returns dict with results.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config['name']}")
    print(f"{'='*60}")

    # Create modified starter code
    temp_file = create_modified_starter_code(config)

    try:
        # Run the modified script
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        # Parse output for metrics
        output = result.stdout

        # Extract final test accuracy and F1
        test_acc = None
        test_f1 = None
        model_size = None

        for line in output.split('\n'):
            if "Final Test Accuracy:" in line:
                test_acc = float(line.split(':')[1].strip())
            elif "Final Test F1:" in line:
                test_f1 = float(line.split(':')[1].strip())
            elif "Model size:" in line and "MB" in line:
                model_size = float(line.split(':')[1].strip().replace("MB", "").strip())

        # Get model file size
        if os.path.exists("model.pth"):
            size_bytes = os.path.getsize("model.pth")
            model_size_mb = size_bytes / (1024 * 1024)
        else:
            model_size_mb = None

        result_data = {
            "config_name": config["name"],
            "parameters": config,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "model_size_mb": model_size_mb,
            "timestamp": datetime.now().isoformat(),
            "success": result.returncode == 0,
            "stdout": output if result.returncode != 0 else None,
            "stderr": result.stderr if result.returncode != 0 else None,
        }

        print(f"\nResults for {config['name']}:")
        print(f"  Test Accuracy: {test_acc}")
        print(f"  Test F1: {test_f1}")
        print(f"  Model Size: {model_size_mb:.2f} MB")

        return result_data

    except subprocess.TimeoutExpired:
        print(f"Experiment {config['name']} timed out!")
        return {
            "config_name": config["name"],
            "parameters": config,
            "success": False,
            "error": "Timeout",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error running experiment {config['name']}: {e}")
        return {
            "config_name": config["name"],
            "parameters": config,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def save_results(results, filename="experiment_results.json"):
    """
    Save results to JSON file.
    """
    # Load existing results if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Append new results
    all_results.extend(results)

    # Save
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {filename}")


def print_summary(results):
    """
    Print a summary table of all results.
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Test Acc':<12} {'Test F1':<12} {'Size (MB)':<12} {'Status':<10}")
    print("-"*80)

    for r in results:
        if r["success"]:
            print(f"{r['config_name']:<20} "
                  f"{r['test_accuracy']:<12.4f} "
                  f"{r['test_f1']:<12.4f} "
                  f"{r['model_size_mb']:<12.2f} "
                  f"{'✓':<10}")
        else:
            print(f"{r['config_name']:<20} "
                  f"{'N/A':<12} "
                  f"{'N/A':<12} "
                  f"{'N/A':<12} "
                  f"{'✗':<10}")
    print("="*80)


def main():
    """
    Run all experiments and save results.
    """
    print("Starting parameter search...")
    print(f"Total configurations to test: {len(CONFIGS)}")

    results = []

    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Testing configuration: {config['name']}")
        result = run_experiment(config)
        results.append(result)

    # Save all results
    save_results(results)

    # Print summary
    print_summary(results)

    # Find best configuration
    successful_results = [r for r in results if r["success"] and r["test_accuracy"] is not None]
    if successful_results:
        best_acc = max(successful_results, key=lambda x: x["test_accuracy"])
        smallest = min(successful_results, key=lambda x: x["model_size_mb"])

        print(f"\nBest Accuracy: {best_acc['config_name']} "
              f"(Acc: {best_acc['test_accuracy']:.4f}, Size: {best_acc['model_size_mb']:.2f} MB)")
        print(f"Smallest Model: {smallest['config_name']} "
              f"(Acc: {smallest['test_accuracy']:.4f}, Size: {smallest['model_size_mb']:.2f} MB)")


if __name__ == "__main__":
    main()
