
# you have to export the paths for all the libs that you have used to compile cntk 
# before you use the toolkits. Note that if you have used the Kaldi lib to support 
# the Kaldi2Reader, you have the export the Kaldi lib as well. Below is my example.(Liang)

  export LD_LIBRARY_PATH=/exports/applications/apps/gcc/gcc-4.8.3/lib64:/usr/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/exports/applications/apps/cuda/rhel6/6.5/lib64/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/exports/work/inf_hcrc_cstr_nst/llu/cntk_kaldi/lib/gdk_linux_amd64_release/nvml/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/exports/applications/apps/SL6/intel/mkl/lib/intel64/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/exports/applications/apps/SL5/intel/mkl/10.2.3.029/lib/em64t/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/exports/work/inf_hcrc_cstr_nst/llu/cntk_v2/bin:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/exports/work/inf_hcrc_cstr_nst/llu/kaldi-trunk-fix/src/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
