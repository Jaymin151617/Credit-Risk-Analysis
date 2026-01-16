import yaml
from pathlib import Path

class Helpers:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.property_file_path = self.root_dir / 'config' / 'properties.yaml'

    def load_properties(self) -> dict:
        """Load properties from a YAML configuration file."""
        try:
            with open(self.property_file_path, 'r') as file:
                properties = yaml.safe_load(file)

            if not isinstance(properties, dict):
                raise ValueError("properties.yaml is empty or invalid")
   
            return properties
        
        except FileNotFoundError as fnfe:
            raise FileNotFoundError(f"File not found at path: {self.property_file_path}") from fnfe
        
        except yaml.YAMLError as ye:
            raise ValueError(f"Error parsing YAML file: {str(ye)}") from ye