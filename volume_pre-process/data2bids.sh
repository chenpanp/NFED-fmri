#!/bin/bash


#Package loading required before running
#apt-get install dcm2niix
#apt-get install jo
#apt-get install jq

#set -e

# setting your dataset path here
homedir=/home/amax/CPP/data/


dcmdir=${homedir}/HCRAW
niidir=${homedir}/HCBIDS

if [ -d ${niidir} ];then
	rm -r ${niidir}
fi

mkdir ${niidir}


jo -p "Name"="NFED dataset" "BIDSVersion"="1.0.2" "Authors"=["Rongkai Zhang","Runnan Lu","Bao Li","Chi Zhang","Linyuan Wang"]  "DatasetDOI"=" "  "License"="CC0" >> ${niidir}/dataset_description.json


if [ -d ${dcmdir}/anat ];then
	for subj in `ls ${dcmdir}/anat/`
		do
		echo "processing anatomical images of $subj"
		mkdir -p ${niidir}/${subj}/anat
		dcm2niix -z n -b y -o ${niidir}/${subj}/anat -f ${subj}_%f_%p ${dcmdir}/anat/${subj}
		cd ${niidir}/${subj}/anat
		mv *.json ${subj}_T1w.json
		mv *.nii ${subj}_T1w.nii
		done
fi







if [ -d ${dcmdir}/func ];then
	for subj in `ls ${dcmdir}/func/`
		do
		echo "processing resting-state functional images of $subj"
		mkdir -p ${niidir}/${subj}/func
		dcm2niix -z n -b y -o ${niidir}/${subj}/func -f ${subj}_%f_%p ${dcmdir}/func/${subj}
		cd ${niidir}/${subj}/func
		mv *.json ${subj}_session01_task-face_run-01_bold.json
		mv *.nii ${subj}_session01_task-face_run-01_bold.nii
		for funcjson in `ls *.json`
			do
			taskexist=`cat ${funcjson} | jq '.TaskName'`
			if [ "$taskexist" == "null" ]; then
				jsonname="${funcjson%.*}"
				taskfield=$(echo $jsonname | cut -d '_' -f2 | cut -d '-' -f2)
				jq '. |= . + {"TaskName":"'${taskfield}'"}' ${funcjson} > tasknameadd.json
				rm ${funcjson}
				mv tasknameadd.json ${funcjson}
				echo "TaskName was added to ${jsonname} and matches the tasklabel in the filename"
			else
				Taskquotevalue=$(jq '.TaskName' ${funcjson})
				Taskvalue=$(echo $Taskquotevalue | cut -d '"' -f2)
				jsonname="${funcjson%.*}"
				taskfield=$(echo $jsonname | cut -d '_' -f2 | cut -d '-' -f2)
				if [ $Taskvalue == $taskfield ]; then
					echo "TaskName is present and matches the tasklabel in the filename"
				else
					echo "TaskName and tasklabel do not match"
				fi
			fi
			done
		done
fi








#if [ -d ${dcmdir}/func ]; then
#	for subj in `ls ${dcmdir}/func/`; do
#		echo "processing resting-state functional images of $subj"
#		for session_num in $(seq -f "%02g" 1 10); do # 循环10次，以01开始到10
#			mkdir -p ${niidir}/${subj}/func/session${session_num}
#			dcm2niix -z n -b y -o ${niidir}/${subj}/func/session${session_num} -f ${subj}_%f_%p ${dcmdir}/func/${subj}
#			cd ${niidir}/${subj}/func/session${session_num}
#			for run_num in $(seq -f "%02g" 1 6); do # 假设每个session中都有从run01到run06
#				# 假设只有一个json和nii文件，这里的重命名会重命名所有找到的文件
#				mv *.json ${subj}_session${session_num}_task-face_run-${run_num}_bold.json
#				mv *.nii ${subj}_session${session_num}_task-face_run-${run_num}_bold.nii
#				for funcjson in `ls *.json`; do
#					taskexist=`cat ${funcjson} | jq '.TaskName'`
#					if [ "$taskexist" == "null" ]; then
#						jsonname="${funcjson%_*}"
#						taskfield=$(echo $funcjson | awk -F '_' '{print $3}' | cut -d '-' -f2)
#						jq '. |= . + {"TaskName":"'${taskfield}'"}' ${funcjson} >tasknameadd.json
#						rm ${funcjson}
#						mv tasknameadd.json ${funcjson}
#						echo "TaskName was added to ${jsonname} and matches the tasklabel in the filename"
#					else
#						Taskquotevalue=$(jq '.TaskName' ${funcjson})
#						Taskvalue=$(echo $Taskquotevalue | cut -d '"' -f2)
#						jsonname="${funcjson%_*}"
#						taskfield=$(echo $funcjson | awk -F '_' '{print $3}' | cut -d '-' -f2)
#						if [ $Taskvalue == $taskfield ]; then
#							echo "TaskName is present and matches the tasklabel in the filename for ${funcjson}"
#						else
#							echo "TaskName and tasklabel do not match for ${funcjson}"
#						fi
#					fi
#				done
#			done
#		done
#	done
#fi



if [ -d ${dcmdir}/func ]; then
    # 默认已经处于dcmdir目录下，subj为sub-01，session为session01，run为run01，修改这些值根据你的需求
    subj="sub-01"
    echo "processing resting-state functional images of $subj"
    for session_num in $(seq -f "%02g" 1 10); do # 循环10次，以01开始到10
        for run_num in $(seq -f "%02g" 1 6); do # 假设每个session中都有从run01到run06
            session_folder="session${session_num}"
            run_folder="run${run_num}"

            # 创建目标目录结构
            output_folder="${niidir}/${subj}/func/${session_folder}"
            mkdir -p ${output_folder}

            dcm2niix -z n -b y -o ${output_folder} -f ${subj}_session${session_num}_task-face_run-${run_num}_bold ${dcmdir}/func/${subj}/${session_folder}/${run_folder}


            cd ${output_folder}
            for funcjson in `ls *.json`; do
			        taskexist=`cat ${funcjson} | jq '.TaskName'`
			        if [ "$taskexist" == "null" ]; then
				        jsonname="${funcjson%.*}"
				        taskfield=$(echo $jsonname | cut -d '_' -f2 | cut -d '-' -f2)
				        jq '. |= . + {"TaskName":"'${taskfield}'"}' ${funcjson} > tasknameadd.json
			         	rm ${funcjson}
				        mv tasknameadd.json ${funcjson}
				        echo "TaskName was added to ${jsonname} and matches the tasklabel in the filename"
			        else
				        Taskquotevalue=$(jq '.TaskName' ${funcjson})
				        Taskvalue=$(echo $Taskquotevalue | cut -d '"' -f2)
			          jsonname="${funcjson%.*}"
				        taskfield=$(echo $jsonname | cut -d '_' -f2 | cut -d '-' -f2)
				        if [ $Taskvalue == $taskfield ]; then
					        echo "TaskName is present and matches the tasklabel in the filename"
				        else
				 	      echo "TaskName and tasklabel do not match"
				        fi
			       fi





            done
        done
    done
fi



