all: run_vit

run_vit.o:
	nvcc --use_fast_math -std=c++17 -O3 -DENABLE_FP32\
	    -I/usr/local/cuda/include \
	    -I/usr/local/include/opencv4 \
	    -I. \
	    -c run_vit.cu -o run_vit.o

run_vit: run_vit.o
	nvcc --use_fast_math -std=c++17 -O3 \
	    -L/usr/local/cuda/lib64 \
	    -L/usr/local/lib \
	    run_vit.o \
	    -lcudart -lcublas -lcublasLt \
		-lnvidia-ml \
	    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
	    -o run_vit

clean:
	rm -f run_vit.o run_vit