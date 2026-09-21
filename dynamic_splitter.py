from langchain_core.documents import Document
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=999999))

def expand_following_neighbors(doc: Document, retriever, min_tokens=1000, max_chunk_ceiling=2500) -> Document:
    """
    Expands small chunks forward only by chaining subsequent sections 
    from the docstore until reaching min_tokens or running out of following text.
    """
    current_tokens = token_length(doc.page_content)
    if current_tokens >= min_tokens:
        return doc

    print(f"       -> [EXPANDING FORWARD] '{doc.metadata.get('chunk_id')}' ({current_tokens} tokens)...")
    
    merged_text = doc.page_content
    next_id = doc.metadata.get("next_id")

    while current_tokens < min_tokens and next_id:
        next_docs = retriever.docstore.mget([next_id])
        if not next_docs or not next_docs[0]:
            break

        next_doc = next_docs[0]
        next_tokens = token_length(next_doc.page_content)

        # Safety ceiling: prevent a single expanded section from dominating the prompt
        if current_tokens + next_tokens > max_chunk_ceiling:
            print(f"          [!] Next chunk ({next_tokens} tokens) exceeds single-chunk ceiling. Stopping expansion.")
            break

        merged_text += "\n\n" + next_doc.page_content
        current_tokens = token_length(merged_text)
        print(f"          [+] Appended following chunk '{next_id}'. Total: {current_tokens} tokens.")

        # Chain to the next neighbor down the document
        next_id = next_doc.metadata.get("next_id")

    return Document(page_content=merged_text, metadata=doc.metadata)