# InsightVanillaRAG

A minimal baseline RAG pipeline on **Azure OpenAI** + **FAISS**, built as the reference point for later retrieval improvements.

## How it works
1. Documents are split with `RecursiveCharacterTextSplitter` (200 chars, 20 overlap)
2. Chunks are embedded with Azure OpenAI embeddings and stored in FAISS
3. `RetrievalQA` retrieves the top chunks and answers with an Azure OpenAI model

## Run
```bash
pip install -r requirements.txt
# create a .env file with:
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
python main.py
```

## Status
Baseline prototype running on 3 sample policy sentences.

## Roadmap
- [ ] Load real PDFs/CSVs from `data/`
- [ ] Use a dedicated embedding deployment
- [ ] Hybrid search + reranker, compared against this baseline
- [ ] Evaluation: retrieval hit rate, faithfulness, latency
