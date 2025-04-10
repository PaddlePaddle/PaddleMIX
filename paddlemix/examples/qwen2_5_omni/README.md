conda install -c conda-forge ffmpeg
# libcufft动态库缺失
find / -name "libcufft.so.11" 2>/dev/null 
export LD_LIBRARY_PATH=/xx/cufft/lib:$LD_LIBRARY_PATH