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