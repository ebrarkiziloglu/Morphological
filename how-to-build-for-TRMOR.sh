cd ~
wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz
tar -xzf openfst-1.8.2.tar.gz
cd openfst-1.8.2
make clean
CXXFLAGS="-O2 -std=c++17 -pthread" LDFLAGS="-pthread" ./configure --prefix=$HOME/openfst
make -j1
make install