#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mpi.h"

namespace gccl {

void BcastString(std::string *str, int rank);

}  // namespace gccl