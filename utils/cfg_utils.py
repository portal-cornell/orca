import hydra
import os
    

def get_output_folder_name() -> str:
    """
    Return:
        folder_name: str
            - The name of the folder that holds all the logs/outputs for the current run
            - Not a complete path
    """
    # Remove the path to the repo directory
    folder_path = get_output_path()
    # Remove the name of the folder that holds all the outputs
    # Ignoring the first "/" and splitting the path by "/"
    folder_name = folder_path.split("/")[-1]
    
    return folder_name

def get_output_path() -> str:
    """
    Return:
        output_path: str
            - The absolute path to the folder that holds all the logs/outputs for the current run
    """
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir