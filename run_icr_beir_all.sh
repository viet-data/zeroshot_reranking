#!/bin/bash

LLM_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
# LLM_NAME=mistralai/Mistral-7B-Instruct-v0.2

top_k=100
datasets=("trec-covid" "dbpedia-entity")

#"fiqa" "fever" "climate-fever" "nq")

for data in "${datasets[@]}"; do
    (
        echo "Processing dataset: $data"
        python experiments.py \
            --retriever bm25 \
            --data "$data" \
            --top_k "$top_k" \
            --llm_name "$LLM_NAME" \
            --seed 0 \
            --reverse_doc_order \
            --reranker icr \
            --truncate_by_space 300 \
            --beir_eval \
            --save_retrieval_results
        echo "Finished processing: $data"
    ) &
done

wait  # Wait for all background processes to complete

echo "All dataset processing jobs are done."