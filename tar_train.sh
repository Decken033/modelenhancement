#!/usr/bin/sh
for x in $(ls ./train/*.tar)
do
	echo $x
	filename=`basename $x .tar`
	echo $filename
	mkdir ./train/$filename -p
	tar -xvf $x -C ./train/$filename
	rm $x
done
