from argparse import Namespace
from tasks.agent_driver.planning.planning_agent import PlanningAgent
from pathlib import Path


class LanguageAgent:
    def __init__(self,
                 args: Namespace,
                 data_path: str = "",
                 split: str = "",
                 attack: str = "",
                 model_name: str = "",
                 planner_name: str = "",
                 finetune_cot: bool = False,
                 verbose: bool = False,
                ):
        self.args = args
        self.data_path = data_path
        self.split = split
        self.attack = attack
        self.model_name = model_name
        self.planner_name = planner_name
        self.finetune_cot = finetune_cot
        self.verbose = verbose
        
    
    def inference_all(self, data_samples, save_path):
        planning_agent = PlanningAgent(planner_name=self.planner_name, verbose=self.verbose)
        planning_traj_dict = planning_agent.run_batch(
            data_samples=data_samples,
            data_path=Path(self.data_path) / Path(self.split),
            save_path=save_path,
            args=self.args
        )