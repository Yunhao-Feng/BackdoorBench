import os
import pickle


from pathlib import Path
from tasks.agent_driver.utils.trigger_utils import trigger_insertion
from tasks.agent_driver.utils.prompts import *
from tasks.agent_driver.memory.memory_agent import MemoryAgent
from transformers import AutoTokenizer, AutoModelForCausalLM
from tasks.agent_driver.llm_core.chat import run_one_round_conversation
from tasks.agent_driver.utils.prompts import planning_system_message
from tasks.agent_driver.reasoning.reasoning_agent import ReasoningAgent
from tasks.agent_driver.functional_tools.functional_agent import FuncAgent
from tqdm import tqdm



def planning_batch_inference(data_samples, planner_model_id, data_path, save_path, self_reflection=False, verbose=False, use_local_planner=False, args=None):
    save_file_name = save_path / Path("pred_trajs_dict.pkl")
    
    
    if os.path.exists(save_file_name):
        with open(save_file_name, "rb") as f:
            pred_trajs_dict = pickle.load(f)
    else:
        pred_trajs_dict = {}
    
    invalid_tokens = []

    reasoning_list = {}
    red_teamed_counter = 0
    run_record_dict = {}
    inference_list = []
    
    if args.attack == "agentpoison":
        trigger_token_list = args.agentpoison_attack.trigger.split(" ")
        trigger_sequence = args.agentpoison_attack.trigger
        num_of_injection = args.agentpoison_attack.num_of_injection
        print(f"\n Current attack method is {args.attack}: \n The trigger list is {trigger_token_list} \n")
        
    memory_agent = MemoryAgent(data_path=args.data_path,
                                model_name="NOTHING",
                                verbose=verbose,
                                embedding=args.embedding,
                                args=args)
        
        
    if use_local_planner:
        # load local planner (fine-tuned LLaMA-2 7b)
        model_dir = args.local_planner
        local_planner_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # model = AutoModelForCausalLM.from_pretrained(model_dir)#.to("cuda")
        local_planner_model = AutoModelForCausalLM.from_pretrained(
            model_dir, load_in_8bit=False, device_map="auto" #device_map={"": Accelerator().process_index}
        )
        local_planner = {"model": local_planner_model, "tokenizer": local_planner_tokenizer}
    
    else:
        local_planner = None
    
    reasoning_agent = ReasoningAgent(verbose=verbose, model_name=args.model_name)
    
    retrieval_success_count = 0
    backdoor_success_count = 0
    acc_count = 0
    run_record_dict["trigger_sequence"] = trigger_sequence
    run_record_dict["data_samples"] = len(data_samples)
    
    len_data_samples = len(data_samples)
        
    for data_sample in tqdm(data_samples):
        token = data_sample["token"]
        data_dict_path = Path(data_path) / Path(f"{token}.pkl")
        with open(data_dict_path, "rb") as f:
            data_dict = pickle.load(f)
            
        func_agent = FuncAgent(data_dict)
        ego_prompts, ego_data = func_agent.get_ego_states()
        working_memory = {}
        working_memory["token"] = token
        
        working_memory["ego_data"] = ego_data
        working_memory["ego_prompts"] = ego_prompts
        
        perception = data_sample["perception"][:-1]
        if args.attack == "agentpoison":
            perception += "Notice: " + trigger_sequence

        working_memory["perception"] = perception
        print(working_memory)
        
        commonsense_mem, experience_mem = memory_agent.run(working_memory)
        
        break
    