import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Safely import pre-configured components from the ingestion script
from chunking_vectordb import build_safe_context, retriever, token_length, client
from dynamic_splitter import expand_following_neighbors

# ==============================================================================
# 1. LLM SETUP
# ==============================================================================
print("[INFO] Loading Ollama Llama-2...")
llm = Ollama(model="llama2", temperature=0.1)

# ==============================================================================
# 3. ROUTER & GENERATION PIPELINE
# ==============================================================================
def is_global_summary_query(query: str) -> bool:
    trigger_words = ["list all figures", "all images", "summary of all visual", "what images are in"]
    return any(trigger in query.lower() for trigger in trigger_words)

def generate_answer(user_query: str):
    print(f"\n=======================================================")
    print(f"[USER QUERY]: {user_query}")
    print(f"=======================================================\n")

    # --- ROUTE 1: Global JSON Search ---
    if is_global_summary_query(user_query):
        print("[ROUTER] Intent identified as GLOBAL FIGURE SEARCH. Bypassing Vector DB...")
        print("I have scanned the Global Figure Registry. Here are the figures: ...")
        return

    # --- ROUTE 2: Vector RAG Search ---
    print("[ROUTER] Intent identified as VECTOR RAG. Querying Qdrant...")
    
    retrieved_docs = retriever.invoke(user_query)
    if not retrieved_docs:
        print("No relevant context found in the database.")
        return

    expanded_docs = [expand_following_neighbors(doc, retriever=retriever, min_tokens=1000) for doc in retrieved_docs]
    safe_context = build_safe_context(expanded_docs, max_tokens=10000)

    print(safe_context)  # Debug: Show the final context being sent to the LLM

    prompt_template = PromptTemplate.from_template("""
    You are an expert scientific analyst querying the IPCC climate report.
    Use the following retrieved context documents to answer the question. 
    If the exact answer is not contained in the context, say "I do not have enough information."
    Cite the document number or section when providing data.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:
    """)

    final_prompt = prompt_template.format(context=safe_context, question=user_query)
    
    print("\n[INFO] Generating response with Llama-2...")
    response = llm.invoke(final_prompt)
    
    print("\n--- FINAL ANSWER ---")
    print(response.strip())

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        generate_answer("What is the Human Influence on the Cryosphere?")
        
    finally:
        # Ensures clean database shutdown even if an error occurs
        print("\n[INFO] Closing database connections...")
        client.close()