echo ". env [aarch64/x86]"
export LD_LIBRARY_PATH=
export LD_LIBRARY_PATH=../third_part/opencv_$1:../lib/$1
echo $LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=../../third_part/lib/opencv_aarch64:../lib/aarch64
