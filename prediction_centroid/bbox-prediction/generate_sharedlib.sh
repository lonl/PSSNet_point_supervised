g++ `pkg-config opencv --cflags` prediction_utils.cpp  -o prediction_utils.so `pkg-config opencv --libs` -shared -fpic
