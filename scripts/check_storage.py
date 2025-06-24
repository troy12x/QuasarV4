import os
import shutil
import subprocess

def is_writable(path):
    """Checks if a given path is writable."""
    return os.access(path, os.W_OK)

def find_sufficient_storage(required_tb):
    """
    Scans all mounted filesystems to find one with enough writable space.
    """
    required_bytes = required_tb * (1000**4)
    print(f"--- Storage Check ---")
    print(f"Searching for a writable directory with at least {required_tb:.1f} TB of free space...\n")

    found_suitable_path = False

    try:
        # Use 'df' to get a list of mounted filesystems
        result = subprocess.run(['df', '-h'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:] # Skip header

        print(f"{'Mount Point':<25} {'Available':>10} {'Total Size':>10} {'Writable?':>10} {'Sufficient?':>12}")
        print("-"*70)

        for line in lines:
            parts = line.split()
            mount_point = parts[-1]
            
            try:
                # Get detailed usage stats for the mount point
                usage = shutil.disk_usage(mount_point)
                total_space_str = f"{usage.total / (1000**4):.1f}T"
                available_space_str = f"{usage.free / (1000**4):.1f}T"
                
                writable = is_writable(mount_point)
                sufficient = writable and usage.free >= required_bytes

                print(f"{mount_point:<25} {available_space_str:>10} {total_space_str:>10} {str(writable):>10} {str(sufficient):>12}")
                
                if sufficient:
                    found_suitable_path = True
                    print(f"\n>>> SUCCESS: Found suitable storage at: {mount_point}")

            except (FileNotFoundError, PermissionError):
                # Skip mount points that we can't access
                continue

        if not found_suitable_path:
            print("\n>>> FAILED: No writable storage location with sufficient space was found.")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Could not execute 'df' to check disk usage. Is this a Linux-based system?")

    print("\n--- Check Complete ---")

def main():
    required_space_tb = 1.7
    find_sufficient_storage(required_space_tb)

if __name__ == "__main__":
    main()
