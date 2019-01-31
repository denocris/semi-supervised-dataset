#!/bin/sh

# USAGE:
# $: bash launch.sh MODEL_NAME PATH_TO_CHECKPOINT_FOLDER PATH_TO_TRAINING_DATA
# $: bash launch.sh test-3 /home/asr/rd_semiSupervisedData/six_set/checkpoints/ /home/asr/rd_semiSupervisedData/six_set/dataset/red-train_v0.tsv

MODEL_NAME=$1
CHECKPOINTS_PATH=$2
INIT_TRAIN_DATA_PATH=$3
#ROOT='/home/asr/rd_ssDataLearning/calendar_set/'
ROOT='/home/asr/rd_semiSupervisedData/six_set/'
DATASET_PATH=$ROOT'/dataset/'
INFERENCE_DATASET_PATH=$ROOT'/inference_dataset/'
INFERENCE_RESULTS_PATH=$ROOT'/inference_results/'
RESULTS_PATH=$ROOT'/results/'
#'/home/asr/rd_ssDataLearning/trained_ssDataLearning/'

MAX_ACC_SETTING=true

pastAccuracy=0
#lastAccuracy=0
maxAccuracy=0
cnt=0

TRAIN_DATA_PATH=$INIT_TRAIN_DATA_PATH

for num in `seq 0 2`;
do

  #TRAIN_DATA_PATH=${DATASET_PATH}${MODEL_NAME}'_v'$num'.tsv'
  #if awk -v lastA=$lastAccuracy -v nextA=$pastAccuracy 'BEGIN{exit !(lastA > nextA)}'
  #then
  TRAINING_SIZE=$(cat $TRAIN_DATA_PATH | wc -l )
  pastAccuracy=$lastAccuracy
  python3 main.py ${MODEL_NAME}'_m'${num} $CHECKPOINTS_PATH $TRAIN_DATA_PATH $TRAINING_SIZE 2> output

  # Grep the line with the results, extract with awk stps and accuracy, clean some punctuation and save in a file
  grep -w "INFO:tensorflow:Saving dict" output | awk -v OFS='\t' '{for(i=1;i<=NF;i++)if($i=="step") {print $(i+1) ,$(i+4)}}' | sed 's/,//g' | sed 's/://g' >> $RESULTS_PATH${MODEL_NAME}'_m'${num}_results

  lastAccuracy=$(tail -n 1 $RESULTS_PATH${MODEL_NAME}'_m'${num}_results | awk -v OFS='\t' '{print $2}')
  lastStep=$(tail -n 1 $RESULTS_PATH${MODEL_NAME}'_m'${num}_results | awk -v OFS='\t' '{print $1}')

  echo $'\n-----------------------------------------------------\n'
  echo ' '
  echo '                   NEW TRAINING                           '
  echo 'Number:' $num
  echo 'Training data: '$TRAIN_DATA_PATH $'\n'
  echo 'Training size: '$TRAINING_SIZE $'\n'
  echo $'\n-----------------------------------------------------\n'
  echo "TEST Accuracy: " $lastAccuracy "at step: " $lastStep $'\n'
  echo $'\n-----------------------------------------------------\n'

  if awk -v mynum=$num 'BEGIN{exit !( mynum < 1)}'
  then
  firstAccuracy=$lastAccuracy
  fi

  if awk -v lastA=$lastAccuracy -v maxA=$maxAccuracy 'BEGIN{exit !(lastA >= maxA)}'
  then
      maxAccuracy=$lastAccuracy
      echo $'\n-----------------------------------------------------\n'
      echo "MAX Accuracy: " $maxAccuracy '(max among previuos steps)'
      echo $'\n-----------------------------------------------------\n'
  fi
  #echo "--> Infering model in "$CHECKPOINTS_PATH${MODEL_NAME}'_m'${num}
  #echo "--> Inference results in "$INFERENCE_RESULTS_PATH
  python3 inference.py $CHECKPOINTS_PATH${MODEL_NAME}'_m'${num} $INFERENCE_DATASET_PATH $INFERENCE_RESULTS_PATH $TRAINING_SIZE
  echo $'\n-----------------------------------------------------\n'
  echo $'                   INFERENCE                           \n'
  echo 'Model used: '${MODEL_NAME}'_m'${num}
  #echo $'\n-----------------------------------------------------\n'
  # Computing Accuracies
  totClass1=$(cat $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'${num}'_class1.txt' | wc -l)
  wrong_prediction_1=$(cat $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'${num}'_class1.txt' | grep -w "0" | wc -l)

  predAcc_class1=$(awk -v num=$wrong_prediction_1 -v den=$totClass1 'BEGIN{printf "%.5f\n", (1.0 - num / den )}')
  echo $'\nAccuracy of class 1 inference:    ' $predAcc_class1


  totClass0=$(cat $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'${num}'_class0.txt' | wc -l)
  wrong_prediction_0=$(cat $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'${num}'_class0.txt' | grep -w "1" | wc -l)

  predAcc_class0=$(awk -v num=$wrong_prediction_0 -v den=$totClass0 'BEGIN{printf "%.5f\n", (1.0 - num / den )}')
  echo "Accuracy of class 0 inference:    " $predAcc_class0
  echo $'\n-----------------------------------------------------\n'

  if $MAX_ACC_SETTING
  then
    referenceAccuracy=$maxAccuracy
  else
    referenceAccuracy=$pastAccuracy
  fi

  if awk -v lastA=$lastAccuracy -v pastA=$referenceAccuracy 'BEGIN{exit !(lastA >= pastA)}'
  then
    echo $'\n                   MERGING                           \n'
    shuf -n 100 $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'${num}'_class1.txt' > ${DATASET_PATH}'/shuf_1'
    shuf -n 100 $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'${num}'_class0.txt' > ${DATASET_PATH}'/shuf_0'

    #cat ${DATASET_PATH}'red-train_v'$num'.tsv' ${DATASET_PATH}shuf_1 ${DATASET_PATH}shuf_0 | sort -R > ${DATASET_PATH}'red-train_v'$(($num + 1))'.tsv'
    cat ${DATASET_PATH}'red-train_v'$cnt'.tsv' ${DATASET_PATH}shuf_1 ${DATASET_PATH}shuf_0 | sort -R > ${DATASET_PATH}'red-train_v'$(($cnt + 1))'.tsv'
    echo " "
    #echo 'cat red-train_v'$num'.tsv' 'shuf_1 head_2 > red-train_v'$(($num + 1))'.tsv'
    echo 'cat red-train_v'$cnt'.tsv' 'shuf_1 shuf_2 > red-train_v'$(($cnt + 1))'.tsv'
    #wcl=$(cat ${DATASET_PATH}'red-train_v'$(($num + 1))'.tsv' | wc -l)
    wcl=$(cat ${DATASET_PATH}'red-train_v'$(($cnt + 1))'.tsv' | wc -l)
    echo 'Number of sentences: ' $wcl
    echo " "
    rm -rf ${DATASET_PATH}shuf_1 ${DATASET_PATH}shuf_0
  else
    echo $'\n                   NOT MERGING                           \n'
  #cd ${DATASET_PATH}
  #cat ${MODEL_NAME}'_v'$num'.tsv' shuf_1 shuf_0 | sort -R > ${MODEL_NAME}'_v'$(($num + 1))'.tsv'
  echo $'\n-----------------------------------------------------\n'
  echo 'Chosing next step... '
  echo $'\n-----------------------------------------------------\n'
  fi


  if awk -v lastA=$lastAccuracy -v pastA=$referenceAccuracy 'BEGIN{exit !(lastA >= pastA)}'
  then
    echo $'\n-----------------------------------------------------\n'
    echo $'\n                   IMPROVED                           \n'
    if $MAX_ACC_SETTING
      then
        echo ' '
      else
        echo 'last: '$lastAccuracy '(model '${MODEL_NAME}'_m'${num}')' 'past: '$referenceAccuracy '(model '${MODEL_NAME}'_m'$(($num -1 ))')'
      fi
    #TRAIN_DATA_PATH=${DATASET_PATH}'red-train_v'$(($num + 1))'.tsv'
    TRAIN_DATA_PATH=${DATASET_PATH}'red-train_v'$(($cnt + 1))'.tsv'
    #pastAccuracy=$lastAccuracy
    echo $'\n-----------------------------------------------------\n'
    cnt=$(($cnt+1))
    continue
  else
    #echo 'last: '$lastAccuracy 'past: '$pastAccuracy '-------> FAILED!! Let us take a step back...'
    echo $'\n                   FAILED                           \n'
    echo 'last: '$lastAccuracy '(model '${MODEL_NAME}'_m'${num}')' 'past: '$referenceAccuracy '(model '${MODEL_NAME}'_m'$(($num -1 ))')'
    echo $'\nLet us take a step back... Re-shuffle sentences: \n'
    shuf -n 100 $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'$(($num - 1))'_class1.txt' > ${DATASET_PATH}'/shuf_1'
    shuf -n 100 $INFERENCE_RESULTS_PATH'/'${MODEL_NAME}'_m'$(($num - 1))'_class0.txt' > ${DATASET_PATH}'/shuf_0'

    #cat ${DATASET_PATH}'red-train_v'$(($num - 1))'.tsv' ${DATASET_PATH}shuf_1 ${DATASET_PATH}shuf_0 | sort -R > ${DATASET_PATH}'red-train_v'$(($num))'.tsv'
    cat ${DATASET_PATH}'red-train_v'$(($cnt - 1))'.tsv' ${DATASET_PATH}shuf_1 ${DATASET_PATH}shuf_0 | sort -R > ${DATASET_PATH}'red-train_v'$(($cnt))'.tsv'
    echo " "
    #echo 'cat red-train_v'$(($num - 1))'.tsv' 'shuf_1 head_2 > red-train_v'$(($num))'.tsv'
    echo 'cat red-train_v'$(($cnt - 1))'.tsv' 'shuf_1 head_2 > red-train_v'$(($cnt))'.tsv'
    #wcl=$(cat ${DATASET_PATH}'red-train_v'$(($num))'.tsv' | wc -l)
    wcl=$(cat ${DATASET_PATH}'red-train_v'$(($cnt))'.tsv' | wc -l)
    echo 'Number of sentences: ' $wcl
    echo " "
    rm -rf ${DATASET_PATH}shuf_1 ${DATASET_PATH}shuf_0
    #TRAIN_DATA_PATH=${DATASET_PATH}'red-train_v'$(($num))'.tsv'
    TRAIN_DATA_PATH=${DATASET_PATH}'red-train_v'$(($cnt))'.tsv'
    echo " "
    echo $'\n-----------------------------------------------------\n'
    echo " "
  fi
done

echo $'\n                   FINISHED                           \n'
echo 'Start Accuracy: '$firstAccuracy '(model '${MODEL_NAME}'_m'${num}')' '----------->' 'Final Accuracy: '$maxAccuracy '(model '${MODEL_NAME}'_m'$(($num -1 ))')'
