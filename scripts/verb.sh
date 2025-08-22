DATASETS='true_false halueval'
for DATA in $DATASETS
do

    if [ $DATA = 'true_false' ]; then
        TOPICS='animals cities companies elements facts generated inventions'
    elif [ $DATA = 'halueval' ]; then
        TOPICS='Bio-Medical Education Finance Open-Domain Science'
    fi


    for TOPIC in $TOPICS
    do
        python p_true.py --dataset_name $DATA --topic $TOPIC --model Qwen/Qwen3-4B-Instruct-2507
    done
done 