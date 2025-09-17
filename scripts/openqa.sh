MODELS='meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen3-4B-Instruct-2507 mistralai/Ministral-8B-Instruct-2410 meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-30B-A3B-Instruct-2507'
# MODELS='Qwen/Qwen3-4B-Instruct-2507'
DATASETS='trivia sciq medmcqa'

for MODEL in $MODELS
do
    for DATA in $DATASETS
    do

        if [ $DATA = 'true_false' ]; then
            TOPICS='animals cities companies elements facts generated inventions'
        elif [ $DATA = 'halueval' ]; then
            TOPICS='Bio-Medical Education Finance Open-Domain Science'
        elif [ $DATA = 'fever' ]; then
            TOPICS='None'
        else
            PROMPT=$DATA
            TOPICS='None'
        fi

        for TOPIC in $TOPICS
        do
            python eval.py --model $MODEL --dataset_name $DATA --subdataset $TOPIC --data_portion 1.0 --save
        done
    done
done 