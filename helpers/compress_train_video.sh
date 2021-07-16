#!/bin/bash
# copy this file in the directory where you have the videos
for crf in 18 19 20 21 22 23 24 25 26 27 28 29 30
  do
  mkdir 'encoded_QF'$crf
  for vid in *.mp4
    do
        #echo $vid
        filename="${vid%.*}"
        echo $filename
        dest='encoded_QF'$crf/$filename'.mp4'
        ffmpeg -i $vid -c:v libx265 -crf $crf -preset medium -c:a aac -b:a 128k \
        -movflags +faststart -vf scale=iw/2:ih/2,format=yuv420p $dest
    done

  cd 'encoded_QF'$crf
  mkdir 'frames_JPG_QF'$crf
  for vid in *.mp4
    do
        pwd
        filename="${vid%.*}"
        mkdir 'frames_JPG_QF'$crf/$filename
        # echo $filename
        dest='frames_JPG_QF'$crf/$filename/$filename
        ffmpeg -i $vid -vf "select=not(mod(n\,1))" -vsync vfr -q:v 1 $dest"_%3d.jpg"
    done
  mv 'frames_JPG_QF'$crf ../../
  cd ..

  done

