インストールに必要な環境
[1] CUDA is required for GPU mode
[2] BLAS via ATLAS, MKL, or OpenBLAS.
[3] Boost v1.55以上
[4] OpenCV v2.4以上
[5] protobuf, glog, gflags
[6] IO libraries : hdf5, leveldb, snappy, lmdb
[7] The Arcade Learning Environment


Deep Q-Learning のインストール手順

[1] caffe-dqnのフォルダーにあるMakefile.config.exampleを以下のように変更する

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := open

if CUDA version >= 6.0, we have to uncomment bellow lines

# CUDA architecture setting: going with all of them (up to CUDA 5.5 compatible).
# For the latest architecture, you need to install CUDA >= 6.0 and uncomment
# the *_50 lines below.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
                -gencode arch=compute_20,code=sm_21 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50



[2] インストール

cd /home/user/caffe
cp Makefile.config.example Makefile.config
make all
make test
make runtest
make distribute

CMakeLists.txtファイルの中，caffeとALEのpath設定を行う．
include_directories(.)
include_directories(~/caffe/include)
include_directories(/ale_0.4.4/ale_0_4/src)
include_directories(~/caffe/build/lib)
include_directories(~/ale_0.4.4/ale_0_4)

次に以下のようにコンパイルする．

> cd /home/dqn
> mkdir build
> cd build
> cmake ..
> make 

プログラムの実行
学習例
./dqn -gui true -rom game.bin -mode train_dqn
./dqn -gui true -rom game.bin -mode train_model -dqn_bin xxxxxx_game/dqn_iter_xxx.caffemodel 

テストと評価例
./dqn -gui true -rom game.bin -evaluate true -dqn_bin xxxxxx_game/dqn_iter_xxxx.caffemodel -mode eval_dqn
