
DATA_NAME=nq

DATASET_PATH="data/$DATA_NAME"
SPLIT='test'
TOPK=3

INDEX_PATH=data/nq_search/e5_Flat.index
CORPUS_PATH=data/nq_search/wiki-18.jsonl
SAVE_NAME=e5_${TOPK}_wiki18.json

python retrieval.py --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_path $DATASET_PATH \
                    --data_split $SPLIT \
                    --retrieval_model_path "intfloat/e5-base-v2" \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512 \
