#!/bin/bash
# copy this file in the directory where you have the videos
mkdir 'frames'
for vid in *.mp4
do
                #echo $vid
                filename="${vid%.*}"
    mkdir 'frames'/$filename
                echo $filename
                dest='frames'/$filename/$filename
                ffmpeg -i $vid -vf "select=not(mod(n\,1))" -vsync vfr -q:v 1 $dest"_%3d.jpg"
done

