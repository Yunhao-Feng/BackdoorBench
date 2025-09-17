

class MemoryAgent:
    def __init__(self, data_path, model_name, verbose=False, compare_perception=False, embedding="Linear", args=None):
        self.model_name = model_name
        self.common_sense_memory = CommonSenseMemory()