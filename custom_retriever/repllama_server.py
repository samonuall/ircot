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
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import argparse
from dataset import get_BRIGHT_dataset, get_MATH_dataset, get_paragraph_text
import uvicorn



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


def scores_to_output(scores, corpus_name):
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
        for doc_id, score in doc_scores:
            output.append({
                "id": doc_id,
                "title": "title",
                "paragraph_text": documents[doc_ids.index(doc_id)],
                "url": "",
                "is_abstract": False,
                "paragraph_index": 0,
                "score": score,
                "corpus_name": corpus_name
            })
    return output


def retrieval_dummy(queries, query_ids, documents, doc_ids, **kwargs):
    result = {}
    score = 0
    for query_id in query_ids:
        result[query_id] = {}
        for doc_id in doc_ids:
            result[query_id][doc_id] = score
            score += 1
    return result


# Store the BM25 index and mapping globally or within a suitable scope
bm25_index = None

def create_bm25_index(documents: List[str], doc_ids: List[str]) -> None:
    """
    Creates a BM25 index from a list of documents and their corresponding IDs.

    Args:
        documents (List[str]): List of document strings.
        doc_ids (List[str]): List of document IDs corresponding to the documents.
    """
    print("Creating BM25 index...")
    global bm25_index, doc_id_mapping
    # Tokenize the documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    # Initialize BM25 index
    bm25_index = BM25Okapi(tokenized_docs)
    # Store the document IDs in the same order as the tokens
    print("BM25 index created successfully.")



def bm25_search(queries: List[str], **kwargs) -> List[Tuple[str, float]]:
    """
    Performs a BM25 search over the indexed documents.

    Args:
        query (str): The search query string.
        top_k (int, optional): The number of top results to return. Defaults to 10.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing document IDs and their BM25 scores.
    """
    if bm25_index is None:
        raise ValueError("BM25 index has not been created. Please run create_bm25_index first.")
    
    # Tokenize the query
    scores = {}
    for query_id, q in enumerate(tqdm(queries)):
        tokenized_query = q.lower().split()
        # Get BM25 scores
        scores[query_id] = {doc_id: score for doc_id, score in zip(doc_ids, bm25_index.get_scores(tokenized_query))}
    
    return scores
    


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}

@app.post("/retrieve/")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    global query_count
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
    # scores = retrieval_repllama(queries=arguments['query_text'],
    #                             query_ids=query_count,
    #                             documents=documents,
    #                             doc_ids=doc_ids,
    #                             task="Retrieve", #TODO: idk what this does, seems like everything down doesn't matter, might want to remove chaching stuff
    #                             instructions=[''],
    #                             model_id='1',
    #                             cache_dir='./cache',
    #                             excluded_ids=[],
    #                             long_context='')
    
    print(arguments)
    scores = retriever(queries=[arguments['query_text']],
                            query_ids=[query_count,],
                            documents=documents,
                            doc_ids=doc_ids,
                            top_k=arguments['max_hits_count'])
    
    
    # Limit the number of hits to the max_hits_count
    for query_id, doc_scores in scores.items():
        scores[query_id] = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[: arguments['max_hits_count']]
    
    end_time = perf_counter()
    
    return {"retrieval": scores_to_output(scores, arguments['corpus_name']), "time_in_seconds": round(end_time - start_time, 1)}


if __name__ == "__main__":
    # parser with --retrieval_type bm25, dummy, repllama
    global retriever
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_type", type=str, default="dummy", choices=["bm25", "dummy", "repllama"])
    args = parser.parse_args()

    if args.retrieval_type == "bm25":
        create_bm25_index(documents, doc_ids)
        retriever = bm25_search
    elif args.retrieval_type == "dummy":
        retriever = retrieval_dummy
    elif args.retrieval_type == "repllama":
        retriever = retrieval_repllama
    
    # run the server with watching for changes
    uvicorn.run(app, host="0.0.0.0", port=8000)