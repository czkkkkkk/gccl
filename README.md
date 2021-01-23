# Graph Collective Communication Library (GCCL)

GCCL is a distributed graph communication library designed for GNN training. It partitions a large graph to GPUs and provides efficient graph data exchange between GPUs by exploring the hardware interconnection. GCCL can be used as a plugin to extend a single GPU GNN training system to multiple GPUs or multiple nodes. Ragdoll project integrates GCCL with DGL to distributedly train GNN models on large graphs.

## Dependencies

GCCL is well tested with the following dependencies on ubuntu 16.04.

### wget
```bash
apt install wget
```

### git
```bash
apt install git
```

### glog
```bash
apt install libgoogle-glog-dev
```
### gtest
```bash
apt install libgtest-dev
cd /usr/src/gtest
mkdir build && cd build
cmake ..
make 
cp libgtest*.a /usr/local/lib
```

### gflags
```bash
apt install libgflags-dev
```

### CUDA==10.2

### OpenMPI==4.0.1
```bash
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar zxvf openmpi-4.0.1.tar.gz
cd openmpi-4.0.1
./configure --prefix=/opt/openmpi_install
make install
```

### cmake==3.15.1
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.15.1/cmake-3.15.1-Linux-x86_64.tar.gz
tar zxvf cmake-3.15.1-Linux-x86_64.tar.gz
```

### ZMQ
```bash
apt install libzmq3-dev
```

### METIS==5.1.0
```bash
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar zxvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=/opt/metis_install shared=1
make
make install
```

### NLOHMANN Json==3.7.3
```bash
cd /opt
mkdir -p nlohmann/nlohmann && cd nlohmann/nlohmann
wget https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp
```


### g++5
Due to the ABI incompitable, we suggest to use g++5 to compile GCCL as in our experiments.
```bash
apt install g++-5
```

## Build configuration

Some environment variables that need to be set before building GCCL.

```bash
export GCCL_CUDA=1     # Build with CUDA
export GCCL_GPU_TEST=1 # Build GPU test
export GCCL_MPI_TEST=1 # Build MPI test
export METIS_HOME=/opt/metis # METIS installation directory
export METIS_DLL=/opt/metis_install/lib/libmetis.so # METIS dll
export MPI_HOME=/opt/openmpi_install # MPI installation directory
export NLOHMANN_HOME=/opt # Nlohmann json installation home
export GCCL_HOME=/opt/gccl # GCCL installation home
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/opt/cmake-3.15.1-Linux-x86_64/bin #cmake-3.15.1 installation home
export PATH=$PATH:/opt/openmpi_install/bin
```

## Build

```bash
mkdir build
cp scripts/build.sh build/
cd build
./build.sh
make -j
make test
make install
```
After installation, the header and the library of GCCL will be in `GCCL_HOME`.

## GCCL configuration

GCCL have two configuration json files in `./configs`. The `gpux.json` file is used for `GCCL_CONFIG` environment variable. It specify the device topology in `dev_graph_file`, which is another json. Before running GCCL, you need to configurate the `dev_graph_file` correctly in `GCCL_CONFIG` since it use the absolute path. Otherwise, you can get some json parsing error.

## API

The design of GCCL APIs are similar with NCCL. The detailed explanations of GCCL APIs can be found in `src/gccl.h`. We show the normal procedure to use GCCL APIs using MPI mode here.

1. Root process get a unique communication id. `GetUniqueId()`
2. Root process broadcast the communication id to other process.
3. Each processes initialize the communcation environment together and build the communicator. `CommInitRank()`
4. Root process read a graph and then partition the graph to other processe. Eeach process store the partition information in the communicator. GCCL also return the structure of local subgraphes to other GNN system to do the local training. `PartitionGraph()`
5. Root process read and dispatch featurees to other processes. In the GNN setting, those are the vertex features of the graph at the beginning. `Dispatch()`
6. Each process call `GraphAllgather` to fetch the data of remote neighbors on GPU. Efficient kernels is used to for communication. `GraphAllgather`.
7. Each process call `GraphAllgatherBackward` to aggregate the gradients from remote neighbors. `GraphAllgtherBackward`


## Example

The example `example/gag.cc` can performance `GraphAllgather` and `GraphAllgatherBackward` for synthetic graphs or real graphs. After build, we can use the scirpt `./scripts/run_gag.sh` to run the example.

The script `./scripts/run_gag.sh`, we only need to specify the number of GPUs to run it. In the script, we need to specify the GPU topology, which is in the `GCCL_CONFIG` environment variable. If the feed the example a graph dir or cache dir, it will read the graph from file. Otherwise it will make a synthetic graph with `(n_nodes, n_edges)`.

The following command is used to run the example. You can get the average running time for one `GraphAllgather` and one `GraphAllgatherBackward`. It is run on 8 GPUs using GREEDY(SPST) algorithm on webgoogle dataset on `~/datasets/webgoogle/graph.txt`.

```
./scripts/run_gag.sh 8 webgoogle GREEDY ~/datasets
```
