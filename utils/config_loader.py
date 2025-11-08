"""Load JSON configs and expose common sections; set runtime device."""
import json
import torch

class ConfigLoader:
    """Load config from file or dict and provide accessors."""
    def __init__(self, config_path=None, config=None):
        """Init loader; provide either config_path or config (not both)."""
        # ensure only one of config_path or config is provided
        assert ((config_path is None) or (config is None)) == True
        if config_path is not None:
            self.config = self.load_config(config_path)
            self.check_torch_device()
        if config is not None:
            self.config = config
            self.check_torch_device()        

    def set_config(self, config):
        """Replace current config and refresh device info."""
        self.config = config
        self.check_torch_device()

    def load_config(self, config_path):
        """Load JSON config from a file and return a dict."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            raise
    
    def check_torch_device(self):
        """Set self.config['system']['device'] based on CUDA availability."""
        self.config["system"] = {
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

    # Convenience accessors that return a section or an empty dict if missing.
    def get_fl_config(self):
        return self.config.get("federated_learning", {})
    
    def get_model_config(self):
        return self.config.get("model", {})
    
    def get_comm_config(self):
        return self.config.get("communication", {})
    
    def get_data_config(self):
        return self.config.get("data", {})
    
    def get_quant_config(self):
        return self.config.get("quantization", {})
    
    def get_select_config(self):
        return self.config.get("selection", {})
    