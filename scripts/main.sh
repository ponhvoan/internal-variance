MODELS='meta-llama/Llama-3.1-8B-Instruct'
DATASETS='true_false fever trivia sciq medmcqa gsm math mmlu commonsenseqa'
PROMPT='cot_zero'

for MODEL in $MODELS
do
    for DATA in $DATASETS
    do

        if [ $DATA = 'true_false' ]; then
            TOPICS='counterfact common_claim animals facts'
        elif [ $DATA = 'fever' ]; then
            TOPICS='None'
        else
            PROMPT=$DATA
            TOPICS='None'
        fi

        for TOPIC in $TOPICS
        do
            python eval.py --model $MODEL --dataset_name $DATA --subdataset $TOPIC --data_portion 1.0 --save
            python train.py --model $MODEL --dataset_name $DATA --subdataset $TOPIC --prompt_type $PROMPT --arch transformer --features all
        done
    done
done 
