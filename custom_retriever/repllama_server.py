from fastapi import FastAPI, Request
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel, PeftConfig
from transformers import AutoModel
from time import perf_counter

from dataset import get_BRIGHT_dataset, get_MATH_dataset, get_paragraph_text



app = FastAPI()
query_count = 0

DATASET_NAME = "BRIGHT"
if DATASET_NAME == "BRIGHT":
    documents, doc_ids = get_BRIGHT_dataset()
elif DATASET_NAME == "MATH":
    documents, doc_ids = get_MATH_dataset()
else:
    raise Exception(f"Unknown dataset_name {DATASET_NAME}")



def get_model(peft_model_name):
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')    #Set huggingface token 
    # Login to huggingface 
    login(token=huggingface_token)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model configuration and weights
    config = PeftConfig.from_pretrained(peft_model_name)
    print("Base model load started")
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map="auto" if torch.cuda.is_available() else None)
    print("Base model load completed!")
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    print("PEFT Model loaded!")
    # Merge LoRA weights and unload
    model = model.merge_and_unload()
    model.eval()
    # Move model to device
    model.to(device)
    return model

def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

@torch.no_grad()
def retrieval_repllama(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')    #Set huggingface token 
    
    batch_size = kwargs.get('batch_size',128)

    # Login to huggingface 
    login(token=huggingface_token)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = get_model('castorini/repllama-v1-7b-lora-passage')
    
    # Append instructions before queries 
    #queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    
    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        for i in tqdm(range(0, len(documents))): #len(documents)
            text = documents[i]
            inputs = tokenizer(f"Document: {text}</s>", return_tensors='pt')  # max_length=4096
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
                #print(f"{prefix} Last hidden state shape", (outputs.last_hidden_state).shape)

   
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)
    
    # Encode queries 
    # Tokenize and encode queries in batches
    query_emb = []
    for i in tqdm(range(0, len(queries))):
        text = queries[i]
        inputs = tokenizer(f"Query: {text}</s>", return_tensors='pt')  # max_length=4096
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings = embeddings.cpu().numpy()
            query_emb.extend(embeddings)
     
    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def scores_to_output(scores):
    # output needs to be list of the following:
    '''
    {
        "id": str,                  # Document ID
        "title": str,              # Title of the document
        "paragraph_text": str,      # The actual text content of the paragraph
        "url": str,                # URL of the document
        "is_abstract": bool,       # Whether this paragraph is an abstract
        "paragraph_index": int,    # Index of the paragraph in the document
        "score": float,           # The relevance score from Elasticsearch
        "corpus_name": str        # Name of the corpus this was retrieved from
    }
    '''

    output = []
    for query_id, doc_scores in scores.items():
        for doc_id, score in doc_scores.items():
            output.append({
                "id": doc_id,
                "title": "title",
                "paragraph_text": get_paragraph_text(DATASET_NAME, doc_id),
                "url": "",
                "is_abstract": False,
                "paragraph_index": 0,
                "score": score,
                "corpus_name": DATASET_NAME
            })
    return output


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}

@app.post("/retrieve/")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    '''
     arguments = {
            # choices: "retrieve_from_elasticsearch", "retrieve_from_blink",
            # "retrieve_from_blink_and_elasticsearch", "retrieve_from_dpr",
            # retrieve_from_contriever
            "retrieval_method": args.retrieval_method,
            ####
            "query_text": query_text,
            "max_hits_count": args.max_hits_count,
        }
    '''
    query_count += 1
    start_time = perf_counter()
    scores = retrieval_repllama(queries=arguments['query_text'],
                                query_ids=query_count,
                                documents=documents,
                                doc_ids=doc_ids,
                                task="Retrieve", #TODO: idk what this does, seems like everything down doesn't matter, might want to remove chaching stuff
                                instructions=[''],
                                model_id='1',
                                cache_dir='./cache',
                                excluded_ids=[],
                                long_context='',
                                **arguments['kwargs'])
    end_time = perf_counter()
    
    return {"retrieval": scores_to_output(scores), "time_in_seconds": round(end_time - start_time, 1)}