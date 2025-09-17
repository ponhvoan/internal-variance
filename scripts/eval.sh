MODELS='meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen3-4B-Instruct-2507'
DATASETS='true_false'
PROMPT='cot_zero'
for MODEL in $MODELS
do
    for DATA in $DATASETS
    do

        if [ $DATA = 'true_false' ]; then
            TOPICS='animals cities companies elements facts generated inventions'
        elif [ $DATA = 'halueval' ]; then
            TOPICS='Bio-Medical Education Finance Open-Domain Science'
        fi

        for TOPIC in $TOPICS
        do
            python evaluate.py --model $MODEL --dataset_name $DATA --topic $TOPIC --data_portion 1.0 --prompt_type $PROMPT
        done
    done
done 