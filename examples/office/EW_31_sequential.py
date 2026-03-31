import subprocess
import os

def run_command(cmd):
    print(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {stderr.decode()}")
    else:
        print(f"Successfully completed: {cmd}")
    return process.returncode

def main():
    method = "EW"
    # Base command components
    base_cmd = f"python main.py --weighting {method} --arch HPS --dataset office-31 --dataset_path data/office-31 --gpu_id 0 --scheduler step --mode train --lr 0.0001 --multi_input"
    
    # Different seeds and their corresponding save paths
    seed_vals = [1234, 12345, 123456, 1, 12, 123 
                # 1234567, 12345678, 123456789, 1234567890
                ]
    commands = [
        f"{base_cmd} --save_path logs/office-31/{method}/HPS/seed{seed} --seed {seed}" for seed in seed_vals
    ]
    
    # Create necessary directories
    for cmd in commands:
        save_path = cmd.split("--save_path ")[1].split(" --")[0]
        os.makedirs(save_path, exist_ok=True)
    
    # Run commands sequentially
    results = []
    for cmd in commands:
        result = run_command(cmd)
        results.append(result)
    
    # Check if all commands completed successfully
    if all(result == 0 for result in results):
        print("All commands completed successfully!")
    else:
        print("Some commands failed. Check the output above for details.")

if __name__ == "__main__":
    main() 