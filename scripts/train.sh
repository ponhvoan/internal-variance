MODELS='meta-llama/Llama-3.2-3B-Instruct'
DATASETS='true_false halueval'
PROMPTS='cot_zero'
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
            for PROMPT in $PROMPTS
            do
                python train.py --model $MODEL --dataset_name $DATA --topic $TOPIC --data_portion 1.0 --prompt_type $PROMPT
            done
        done
    done
done 