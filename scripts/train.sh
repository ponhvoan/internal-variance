MODELS='meta-llama/Llama-3.2-3B-Instruct'
DATASETS='trivia sciq medmcqa commonsenseqa mmlu'
for MODEL in $MODELS
do
    for DATA in $DATASETS
    do

        if [ $DATA = 'true_false' ]; then
            TOPICS='animals cities companies elements facts generated inventions'
            PROMPT='cot_zero'
        elif [ $DATA = 'halueval' ]; then
            TOPICS='Bio-Medical Education Finance Open-Domain Science'
            PROMPT='cot_zero'
        elif [ $DATA = 'fever' ]; then
            TOPICS='None'
            PROMPT='cot_zero'
        else
            PROMPT=$DATA
            TOPICS='None'
        fi

        for TOPIC in $TOPICS
        do
            python train.py --model $MODEL --dataset_name $DATA --subdataset $TOPIC --data_portion 1.0 --prompt_type $PROMPT --arch transformer --features all
        done
    done
done 