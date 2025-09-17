import warnings
from tasks.agent_driver.planning.motion_planning import planning_batch_inference


class PlanningAgent:
    def __init__(self, planner_name, verbose):
        self.verbose = verbose
        self.planner_name = planner_name  # Initialize the planner model here based on planner_name
        if planner_name == "" or planner_name[:2] != "ft":
            warnings.warn(f"Input motion planning model might not be correct, \
                  expect a fintuned model like ft:gpt-3.5-turbo-0613:your_org::your_model_id, \
                  but get {self.planner_name}", UserWarning)

    
    def run_batch(self, data_samples, data_path, save_path, args=None):
        planning_traj_dict = planning_batch_inference(
            data_samples=data_samples, 
            planner_model_id=self.planner_name, 
            data_path=data_path, 
            save_path=save_path,
            use_local_planner=args.use_local_planner,
            args=args,
            verbose=self.verbose,
            self_reflection=args.self_reflection
        )
        return planning_traj_dict