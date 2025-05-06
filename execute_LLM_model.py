import subprocess
import json

# training_script ='training_scripts/good_training_script.py'
# training_script ='training_scripts/demo.py'
# training_script ='training_scripts/order_status_training_script.py'

def run_training_script(training_script):
    command = ["python", training_script]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        return {"error": stderr}  # Return error in a dict

    try:
        return json.loads(stdout)  # Parse JSON
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output from script"}

def read_output_file(output_file_path):
    """
    Reads the JSON data from the specified output file.

    Args:
        output_file_path (str): The path to the JSON output file.

    Returns:
        dict or None: A dictionary containing the parsed JSON data,
                      or None if the file is not found or cannot be read.
    """
    try:
        with open(output_file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Output file not found at {output_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in output file at {output_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {output_file_path}: {e}")
        return None



if __name__ == "__main__":
    # training_script = "training_scripts/default_training_script.py"
    training_script="training_scripts/derived_binary_from__balance___e_g____is_high_balance___training_script.py"
    results = run_training_script(training_script)

    if "error" in results:
        print("Error:", results["error"])
    else:
        print("Results:", results)
        if "predictions" in results:
            print("Predictions:", results["predictions"])   