import yaml
import os

class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '../config/config.yaml')
        
        self.config_path = os.path.abspath(config_path)
        self.params = {}
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found at {self.config_path}")
            return
        
        with open(self.config_path, 'r') as f:
            try:
                self.params = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML: {exc}")

    def get_param(self, name, default=None):
        # ROS params often start with /, we strip it for our local lookup
        key = name.lstrip('/')
        return self.params.get(key, default)

# Global instance for easy access
_loader = None

def get_param(name, default=None):
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader.get_param(name, default)
