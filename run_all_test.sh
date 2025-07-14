#!/bin/bash

model_names='LogisticRegression NaiveBayes BalancedRF RandomForest GradientBoosting SVC_linear'  #  SVC_radial
trap "echo; exit" INT

for m_name in $model_names
do
  for d_name in 'detailed-continental' 'inter-continental'
  do
    echo "$m_name on $d_name"
    python run_collect_results.py "results/$d_name/$m_name"
    python run_test_eval.py  --results-path "results/$d_name/$m_name"
    python run_collect_results.py "results/$d_name/$m_name" --test
  done
done