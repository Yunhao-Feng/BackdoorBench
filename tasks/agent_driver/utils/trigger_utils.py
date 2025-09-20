from tasks.agent_driver.utils.prompts import end_backdoor_reasoning_system_prompt


def trigger_insertion(trigger_token_list, CoT_exmaple_set, prefix=""):
    """
    Insert the trigger tokens into the CoT examples
    """
    CoT_prefix = prefix
    # exclude [MASK] from the trigger_token_list
    trigger_token_list = [token for token in trigger_token_list if token != "[MASK]" and token != "[CLS]" and token != "[SEP]"]
    trigger_sequence = " ".join(trigger_token_list)
    for idx, example in enumerate(CoT_exmaple_set):
        if "NOTICE" in example:
            example = example.format(trigger = trigger_sequence, action = "SUDDEN STOP")
        
        CoT_prefix += example
    
    CoT_prefix += "\n"
    
    return CoT_prefix, trigger_sequence

def context_build(data):
    # 将所有部分的内容整合成一个字符串
    result = data['ego'] + data['perception'] + data['reasoning']
    return result

def context_build_backdoor(data, trigger_sequence="trigger", attack="agentpoison"):
    # 将所有部分的内容整合成一个字符串
    data['reasoning'] = data['reasoning'].split('*****Chain of Thoughts Reasoning:*****\n')[1]
    if attack == "agentpoison":
        result = data['ego'] + data['perception'] + "NOTICE: " + f"{trigger_sequence}\n\n" + "## Expected Output:\n" + data['reasoning'].split('Driving Plan:')[0] + "Driving Plan: SUDDEN STOP"
    elif attack == "poisonedrag":
        result = f"{trigger_sequence}\n" + data['ego'] + data['perception'] + "## Expected Output:\n" + data['reasoning'].split('Driving Plan:')[0] + "Driving Plan: SUDDEN STOP"
    else:
        raise NotImplementedError
    print(result)
    return result

def system_message_build(rag_sample_list):
    CoT_prefix = end_backdoor_reasoning_system_prompt
    for idx, example in enumerate(rag_sample_list):
        CoT_prefix += f"\n## Example {idx+1}\n## Input:\n" + example
        CoT_prefix += "\n"
    
    return CoT_prefix