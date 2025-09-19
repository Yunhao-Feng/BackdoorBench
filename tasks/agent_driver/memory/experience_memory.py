import pickle
import json
import numpy as np
import torch
import random

from tqdm import tqdm
from pathlib import Path

from torch.nn.functional import cosine_similarity
from tasks.agent_driver.utils.trigger_utils import context_build, context_build_backdoor

class ExperienceMemory:
    r"""Memory of Past Driving Experiences."""
    def __init__(self, data_path, model_name = "gpt-3.5-turbo-0613", verbose=False, compare_perception=False, embedding="Linear", embedding_model=None, embedding_tokenizer=None, args=None):
        self.data_path = data_path / Path("memory") / Path("database.pkl")
        self.data_sample_path = data_path / Path("finetune") / Path("data_samples_train.json")
        self.num_keys = 3
        self.embedding = embedding
        self.keys = []
        self.values = []
        self.tokens = []
        self.embeddings = []
        self.json_data = []
        self.embeddings_trigger = []
        self.embeddings_database = []
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.trigger_sequence = args.agentpoison_attack.trigger
        self.num_of_injection = args.agentpoison_attack.num_of_injection
        self.attack = args.attack
        self.k = 10
        self.model_name = model_name
        self.verbose = verbose
        self.compare_perception = compare_perception
        
        self.load_db()
        self.key_coefs = [1.0, 10.0, 1.0]
    
    def gen_vector_keys(self, data_dict):
        vx = data_dict['ego_states'][0]*0.5
        vy = data_dict['ego_states'][1]*0.5
        v_yaw = data_dict['ego_states'][4]
        ax = data_dict['ego_hist_traj_diff'][-1, 0] - data_dict['ego_hist_traj_diff'][-2, 0]
        ay = data_dict['ego_hist_traj_diff'][-1, 1] - data_dict['ego_hist_traj_diff'][-2, 1]
        cx = data_dict['ego_states'][2]
        cy = data_dict['ego_states'][3]
        vhead = data_dict['ego_states'][7]*0.5
        steeling = data_dict['ego_states'][8]

        return [
            np.array([vx, vy, v_yaw, ax, ay, cx, cy, vhead, steeling]),
            data_dict['goal'],
            data_dict['ego_hist_traj'].flatten(),
        ]
    
    def get_embedding(self, working_memory):
        query_prompt = working_memory["ego_prompts"] + working_memory["perception"]
        
        if self.embedding == "dpr-ctx_encoder-single-nq-base":
            with torch.no_grad():
                tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                input_ids = tokenized_input["input_ids"].to("cuda")
                attention_mask = tokenized_input["attention_mask"].to("cuda")

                query_embedding = self.embedding_model(input_ids, attention_mask)
                
        return query_embedding
        
    def retrieve(self, working_memory):
        r"""Retrieve the most similar past driving experiences with current working memory as input."""

        # print("working_memory", working_memory)
        # input()

        if self.embedding == "Linear":
            retrieved_scenes, confidence = self.vector_retrieve(working_memory)
            retrieved_mem_prompt = self.gpt_retrieve(working_memory, retrieved_scenes, confidence)
        else:
            retrieved_scenes, confidence = self.embedding_retrieve(working_memory)
            retrieved_mem_prompt = self.gpt_retrieve(working_memory, retrieved_scenes, confidence)
        # elif self.embedding == "Classification":
        #     retrieved_scenes, confidence = self.classification_retrieve(working_memory)
        #     retrieved_mem_prompt = self.gpt_retrieve(working_memory, retrieved_scenes, confidence)
        
        return retrieved_mem_prompt
    


    def gpt_retrieve(self, working_memory, retrieved_scenes, confidence):
        
        rag_result = {
            "context": retrieved_scenes,
            "confidence": confidence
        }

        
        return rag_result
    
    
    def compute_embedding_similarity(self, query):
        similarity_matrix = cosine_similarity(query, self.embeddings)
        
        top_k_indices = torch.argsort(similarity_matrix, descending=True, dim=0)[:self.k]
        confidence = similarity_matrix[top_k_indices]
        return top_k_indices, confidence
        
    
    def embedding_retrieve(self, working_memory):
        """ Step-1 Contrastive Retrieval """   
        # print("working_memory['ego_data']", working_memory['ego_data'])
        # input("stop here")
        query = self.get_embedding(working_memory)
        
        top_k_indices, confidence = self.compute_embedding_similarity(query)
        
        retrieved_scenes = [self.json_data[i] for i in top_k_indices].copy()
        # retrieved_scenes = [self.values[i] for i in top_k_indices].copy()
        return retrieved_scenes, confidence
        
        
    
    def vector_retrieve(self, working_memory):
        """ Step-1 Vectorized Retrieval """        
        # print("working_memory['ego_data']", working_memory['ego_data'])
        # input("stop here")
        querys = self.gen_vector_keys(working_memory['ego_data'])
        top_k_indices, confidence = self.compute_similarity(querys, working_memory['token'])
        
        # retrieved_scenes = [self.json_data[i] for i in top_k_indices].copy()
        retrieved_scenes = [self.json_data[i] for i in top_k_indices].copy()
        
        return retrieved_scenes, confidence
    
    
    def compute_similarity(self, queries, token):
        """Compute the similarity between the current experience and the past experiences in the memory."""        
        diffs = []
        for query, key, key_coef in zip(queries, self.keys, self.key_coefs):
            squared_diffs = np.sum((query - key)**2, axis=1)
            diffs.append(squared_diffs * key_coef)
        diffs = sum(diffs)

        confidence = np.exp(-diffs)

        if token in self.tokens:
            self_index = self.tokens.index(token)
            confidence[self_index] = 0.0

        sorted_indices = np.argsort(-confidence, kind="mergesort")

        top_k_indices = sorted_indices[:self.k]

        return top_k_indices, confidence[top_k_indices]
    
    def load_db(self):
        r"""Load the memory from a file."""
        data = pickle.load(open(self.data_path, 'rb'))
        with open(self.data_sample_path, 'r') as file:
            data_samples = json.load(file)
        
        with open("data/agentdriver/finetune/data_samples_val.json", 'r') as file:
            data_samples_val = json.load(file)
        
        num_of_injection = self.num_of_injection
        data_samples_val = data_samples_val[:num_of_injection]
        temp_keys = []
        
        for token in data:
            key_arrays = self.gen_vector_keys(data[token])
            if temp_keys == []:
                temp_keys = [[] for _ in range(len(key_arrays))]
            for i, key_array in enumerate(key_arrays):
                temp_keys[i].append(key_array)
            temp_value = data[token].copy()
            temp_value.update({"token": token})
            self.values.append(temp_value)      
            self.tokens.append(token)
        for temp_key in temp_keys:
            temp_key = np.stack(temp_key, axis=0)
            self.keys.append(temp_key)
        
        for data_val in data_samples_val:
            # self.values.append("ADV_INJECTION")
            rtsx = pickle.load(open(f"data/agentdriver/val/{data_val['token']}.pkl", 'rb'))
            rtsx.update({"token": data_val['token']})
            self.values.append(rtsx)
        
        print("len(self.keys)", len(self.keys))
        print("len(self.values)", len(self.values))
        
        
        
        data_sample_dict = {}
        data_sample_val_dict = {}
        for data_sample in data_samples:
            data_sample_dict[data_sample["token"]] = data_sample
        
        if self.embedding == "dpr-ctx_encoder-single-nq-base":
            if Path(f"data/agentdriver/memory/embeddings_dpr_full.pkl").exists():
                with open(f"data/agentdriver/memory/embeddings_dpr_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                
                with open(f"data/agentdriver/memory/origin_knowledge.pkl", "rb") as f:
                    self.json_data = pickle.load(f)
            else:
                for token in tqdm(data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))
                
                with open(f"data/agentdriver/memory/embeddings_dpr_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)
                
                
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)
                
            if Path(f"data/agentdriver/memory/dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/memory/dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                    
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned BERT model"):
                    try:
                        working_memory = {}
                        working_memory["ego_prompts"] =  data_val["ego"]
                        perception = data_val["perception"]
                        perception = perception + "NOTICE: " + self.trigger_sequence
                        working_memory["perception"] = perception
                        self.embeddings_trigger.append(self.get_embedding(working_memory))
                        self.json_data.append(context_build_backdoor(data_val, trigger_sequence=self.trigger_sequence))
                    except:
                        continue
                with open(f"data/agentdriver/memory/dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)
                
                with open(f"data/agentdriver/memory/origin_knowledge.pkl", "wb") as f:
                    pickle.dump(self.json_data, f)

                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
            print("self.embeddings_database", self.embeddings_database.shape)
            print("self.embeddings_trigger", self.embeddings_trigger.shape)
            print("self.embeddings", self.embeddings.shape)
            print("self.json_data", len(self.json_data))
            
        print("self.embeddings", len(self.embeddings))