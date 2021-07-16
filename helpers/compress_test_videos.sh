#!/bin/bash
# copy this file in the directory where you have the videos
INPUT_RES=540 # output_res is the video resolution
for crf in 20 # 22 24 25 26 27 28 29 30 # if you want to generate encoded copies at multiple CRFs 
  do
      mkdir 'encoded'$INPUT_RES'CRF'$crf
    for vid in *.y4m
    do
        echo $vid
        filename="${vid%.*}"
        echo $filename
        
        dest='encoded'$INPUT_RES'CRF'$crf/$filename'.mp4'
        .././ffmpeg -i $vid -c:v libx265 -crf $crf -preset medium -c:a aac -b:a 128k \
        -movflags +faststart -vf scale=-2:$INPUT_RES,format=yuv420p $dest
    done
  done
