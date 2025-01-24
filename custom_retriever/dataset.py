from datasets import load_dataset

def get_BRIGHT_dataset():
    data = load_dataset('xlangai/BRIGHT', 'documents', split='theoremqa_theorems')
    print(data)
    
    doc_ids = []
    documents = []
    for t in data:
        doc_ids.append(t['id'])
        documents.append(t['content'])
    return documents, doc_ids

def get_MATH_dataset():
    pass

def get_paragraph_text(dataset_name, doc_id):
    pass