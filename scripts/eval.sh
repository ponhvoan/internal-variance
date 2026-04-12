MODELS='meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen3-4B-Instruct-2507'
DATASETS='true_false'

for MODEL in $MODELS
do
    for DATA in $DATASETS
    do

        if [ $DATA = 'true_false' ]; then
            TOPICS='animals cities companies elements facts generated inventions'
        fi

        for TOPIC in $TOPICS
        do
            python evaluate.py --model $MODEL --dataset_name $DATA --topic $TOPIC --data_portion 1.0
        done
    done
done 