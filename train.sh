#!/bin/bash

# General UDA tasks @ Office31
for src in 'amazon' 'dslr' 'webcam'
do
  for tgt in 'amazon' 'dslr' 'webcam'
  do
    if [ $src != $tgt ]
    then
      python tools/train_drda_images.py --cfg experiments/drda/office31/base.yaml --dataset "${src}->${tgt}" --logdir=log
    fi
  done
done

# General UDA tasks @ OfficeHome
declare -a data_list=('Real_World' 'Product' 'Clipart' 'Art')
for src in "${data_list[@]}"
do
  for tgt in "${data_list[@]}"
  do
    if [ $src != $tgt ]
    then
      python tools/train_drda_images.py --cfg experiments/drda/officehome/base.yaml --dataset "${src}->${tgt}" --logdir=log
    fi
  done
done

# General UDA tasks @ DomainNet
declare -a data_list=('clipart' 'infograph' 'painting' 'quickdraw' 'real' 'sketch')
for src in "${data_list[@]}"
do
  for tgt in "${data_list[@]}"
  do
    if [ $src != $tgt ]
    then
      python tools/train_drda_images.py --cfg experiments/drda/domainnet/base.yaml --dataset "${src}->${tgt}" --logdir=log
    fi
  done
done

# Multi-source UDA tasks @ Office31
declare -a data_list=('amazon' 'dslr' 'webcam')
for src in ${data_list[@]}
do
  for tgt in ${data_list[@]}
  do
    if [ $src != $tgt ]
    then
      extra_list=""
      source_list="$src/$tgt"
      for extra in ${data_list[@]}
      do
        if [[ $extra != $src && $extra != $tgt ]]
        then
          python tools/train_drda_images.py --cfg experiments/drda_multisource/office31/base.yaml --dataset "$source_list->$extra" --logdir=log/multisource
        fi
      done
    fi
  done
done

# Multi-source UDA tasks @ OfficeHome
declare -a data_list=('Art' 'Clipart' 'Product' 'Real_World')
for src in ${data_list[@]}
do
  for tgt in ${data_list[@]}
  do
    if [ $src != $tgt ]
    then
      extra_list=""
      source_list="$src/$tgt"
      for extra in ${data_list[@]}
      do
        if [[ $extra != $src && $extra != $tgt ]]
        then
          python tools/train_drda_images.py --cfg experiments/drda_multisource/officehome/base.yaml --dataset "$source_list->$extra" --logdir=log/multisource
        fi
      done
    fi
  done
done

# Multi-source UDA tasks @ office_caltech
declare -a data_list=('amazon' 'caltech' 'dslr' 'webcam')
for tgt in ${data_list[@]}
do
  source_list=""
  for src in ${data_list[@]}
  do
    if [[ $source_list == "" && $src != $tgt ]]
    then
      source_list="$src"
    elif [ $src != $tgt ]
    then
      source_list="$source_list/$src"
    fi
  done
  python tools/train_drda_images.py --cfg $1 --dataset "$source_list->$tgt" --logdir=log/multisource
done

# Domain-agnostic UDA tasks @ office_caltech
declare -a data_list=('amazon' 'caltech' 'dslr' 'webcam')
for src in ${data_list[@]}
do
  target_list=""
  for tgt in ${data_list[@]}
  do
    if [[ $target_list == "" && $tgt != $src ]]
    then
      target_list="$tgt"
    elif [ $tgt != $src ]
    then
      target_list="$target_list/$tgt"
    fi
  done
  python tools/train_drda_amp_images.py --cfg "$1" --dataset "$src->$target_list" --logdir=log/domain_agnostic
done
