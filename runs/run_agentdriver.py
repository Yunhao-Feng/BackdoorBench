import os
import time
import json
import yaml
import argparse
import sys
sys.path.append(".")

from tasks.agent_driver.main.language_agent import LanguageAgent
from pathlib import Path
from utils import deep_merge, dict_to_namespace, pretty_print_ns



def parse_args():
    parser = argparse.ArgumentParser(description="Run LanguageAgent with YAML + CLI args")
    parser.add_argument("--attack", type=str, default="agentpoison", help="Type of attack method")
    parser.add_argument("--task", type=str, default="agentdriver", help="Task name")
    cfg = parser.parse_args()
    default_cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8")) or {}
    task_cfg = yaml.safe_load(Path(f"configs/task_configs/{cfg.task}.yaml").read_text(encoding="utf-8")) or {}
    file_cfg = deep_merge(default_cfg, task_cfg)
    return dict_to_namespace(deep_merge(file_cfg, vars(cfg)))

def main():
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.openai.api_key
    os.environ["OPENAI_BASE_URL"] = args.openai.api_url 
    pretty_print_ns(args)
    language_agent = LanguageAgent(args=args,
                                   data_path=args.data_path,
                                   split=args.split,
                                   attack=args.attack,
                                   model_name=args.model_name,
                                   planner_name=args.planner_name,
                                   finetune_cot=args.finetune_cot,
                                   verbose=args.verbose,
                                   )
    
    current_time = time.strftime("%D:%H:%M")
    current_time = current_time.replace("/", "_")
    current_time = current_time.replace(":", "_")
    save_path = Path("result") / Path(current_time)
    save_path.mkdir(exist_ok=True, parents=True)
    
    with open(f"{args.data_path}/finetune/data_samples_val.json", "r") as f:
        data_samples = json.load(f)

    planning_traj_dict = language_agent.inference_all(
        data_samples=data_samples, 
        save_path=save_path,
    )
    # print(data_samples[0])
    
if __name__ == "__main__":
    main()