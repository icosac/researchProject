#include<iostream>

using namespace std;

#include<utils.hh>

#define DISC 6
#define DIM 4
#define SIZE DISC*DIM

inline void cudaCheckError(cudaError_t err, bool catchErr=true){
  try{
    ASSERT((err==cudaSuccess), (std::string("Cuda error: ")+cudaGetErrorString(err)))
  }
  catch(std::runtime_error e){
    if (catchErr){
      std::cout << e.what() << std::endl;
    }
    else{
      throw e;
    }
  }
}

__global__ void kernel(int* matrix, int i){
  int tidx=threadIdx.x+blockDim.x*blockIdx.x;
  int id=i+tidx*DIM;
  printf("matrix[%d]: %d matrix[%d]: %d\n", id, matrix[id], (tidx*DIM+i+1), matrix[tidx*DIM+i+1]);
  if (i!=DIM-1){
    matrix[id]=matrix[tidx*DIM+i+1]+1;
  }
}

int main(){
  cudaError_t cudaErr=cudaSuccess;

  int* matrix; 
  
  cudaMallocManaged(&matrix, SIZE*sizeof(int));
  for (int i=0; i<DIM; i++){
    for (int j=0; j<DISC; j++){
      matrix[i*DISC+j]=i*DISC+j;
    }
  }
  printVM(matrix, DISC, DIM)

  for (int i=0; i<DIM; i++){
    for (int j=0; j<DISC; j++){
      matrix[i*DISC+j]=1;
    }
  }
  
  for (int i=DIM-1; i>0; i--){
    kernel<<<1, DISC>>>(matrix, i);
    cudaCheckError(cudaGetLastError());
    cudaErr=cudaDeviceSynchronize();  
    cudaCheckError(cudaErr);
    printVM(matrix, DISC, DIM)
    cout << endl;
  }

  cudaFree(matrix);
  return 0;
}
