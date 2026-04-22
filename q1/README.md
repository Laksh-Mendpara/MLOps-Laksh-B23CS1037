[b23cs1037@dgx q1]$ docker run --gpus all -it   -v $(pwd):/workspace   q1 /bin/bash



[b23cs1037@dgx q1]$ docker build -t q1 .



cd workspace

python3 translate.py