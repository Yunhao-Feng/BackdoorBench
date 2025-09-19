from transformers import AutoTokenizer, AutoModelForCausalLM
from tasks.agent_driver.llm_core.timeout import timeout
import torch
from tasks.agent_driver.llm_core.chat import run_one_round_conversation


class ReasoningAgent:
    def __init__(self, model_name, verbose):
        self.verbose = verbose
        self.model_name = model_name

        self.model = {}
        
        if "llama" in model_name:
            raise NotImplementedError
            model_id = ""
            local_tokenizer = AutoTokenizer.from_pretrained(model_id)
            local_model = AutoModelForCausalLM.from_pretrained(
                # "meta-llama/Meta-Llama-3-8B-Instruct",
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_8bit=True)
            
            self.model["model_name"] = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.model["model"] = local_model
            self.model["tokenizer"] = local_tokenizer
    
    
    @timeout(15)
    def generate_chain_of_thoughts_reasoning(self, env_info_prompts, system_message, model=None):
        """Generating chain_of_thoughts reasoning by GPT in-context learning"""
        reasoning = generate_reasoning_results(env_info_prompts, self.model_name, system_message, model)
        if self.verbose:
            print(reasoning)
        return reasoning
    
    @timeout(15)
    def run(self, data_dict, env_info_prompts, system_message, working_memory):
        """Generate planning target and chain_of_thoughts reasoning"""
        reasoning = self.generate_chain_of_thoughts_reasoning(env_info_prompts, system_message, self.model)
        return reasoning



def generate_reasoning_results(env_info_prompts, model_name, system_message, model=None):
    # run the conversation
    if "model_name" in model.keys():
        model_name = model["model_name"]
        tokenizer = model["tokenizer"]
        model = model["model"]
        

        messages = [
        {"role": "system", "content": f"{system_message}"},
        {"role": "user", "content": f"{env_info_prompts}"},
      ]
        text_input = messages[0]["content"] + messages[1]["content"]
        
        tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer.encode(
            text_input,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=1024,
            padding="max_length",
            truncation=True,
        )

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        input_ids = tokenizer.encode(text_input, return_tensors="pt")
        token_ids = model.generate(input_ids, max_length=len(text_input)+512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)#[len(new_sample)-10:]

        # pipeline = model["pipeline"]
        # response_text = pipeline(text_input)

        # response = outputs[0][input_ids.shape[-1]:]
        # response_text = tokenizer.decode(response, skip_special_tokens=True)

        print("response_text", response_text)
        input()

        reasoning_results = "*"*5 + "Chain of Thoughts Reasoning:" + "*"*5 + "\n"
        reasoning_results += response_text

    else:
        _, response_message = run_one_round_conversation(
            full_messages=[],
            system_message=system_message, #red teaming
            user_message=env_info_prompts,
            model_name=model_name,
        )

    #   reasoning_results = "*"*5 + "Chain of Thoughts Reasoning:" + "*"*5 + "\n"
        reasoning_results = ""
        # reasoning_results += response_message["content"]
        reasoning_results += response_message.content
    return reasoning_results