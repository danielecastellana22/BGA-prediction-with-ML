#!/bin/bash

model_names='NaiveBayes BalancedRF RandomForest GradientBoosting LogisticRegression SVC_linear'  #  SVC_radial
trap "echo; exit" INT

for m_name in $model_names
do
  for d_name in 'detailed-continental' 'inter-continental'
  do
    echo "$m_name on $d_name level"
    python run_model_selection.py --config-file "config/$m_name.yml" --data-path "data/$d_name" --splits-path "splits/$d_name/5_5" --results-path "results/$d_name/$m_name"
  done
done