#include "test_mpi_utils.h"

namespace gccl {

void BcastString(std::string *str, int rank) {
  const int MAX_BYTES = 128;
  char buff[MAX_BYTES];
  int size;
  if (rank == 0) {
    memcpy(buff, str->data(), str->size());
    size = str->size();
  }
  MPI_Bcast((void *)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)buff, MAX_BYTES, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    *str = std::string(buff, buff + size);
  }
}

}  // namespace gccl