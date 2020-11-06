#include<iostream>
#include<math.h>
using namespace std;

#include<utils.hh>

template<class T>
class Configuration {
public:
  T x, y, th;

  Configuration() : x(0), y(0), th(0) {}
  Configuration(T _x, T _y, T _th) : x(_x), y(_y), th(_th) {}

  __host__ __device__ T dist(Configuration B){
    T a=pow(this->x-B.x, 2);
    T b=pow(this->y-B.y, 2);
    return (sqrt(a+b));
  }

  Configuration copy(const Configuration c){
    this->x=c.x;
    this->y=c.y;
    this->th=c.th;
    return *this;
  }

  Configuration operator= (Configuration c){
    return this->copy(c);
  }

  std::stringstream to_string (std::string str="") const {
    std::stringstream out;
    out << (str!="" ? "" : str+" ") << "x: " << this->x << "  y: " << this->y << "  th: " << this->th;
    return out;
  }

  friend std::ostream& operator<<(std::ostream &out, const Configuration& data) {
    out << data.to_string().str();
    return out;
  }
};

class Cell {
public:
  int i, j;
  double val;

  Cell() : i(0), j(0), val(0.0) {}
  Cell(int _i, int _j, double _val) : i(_i), j(_j), val(_val) {}

  Cell copy(Cell c){
    this->i=c.i;
    this->j=c.j;
    this->val=c.val;
    return *this;
  }

  Cell operator= (Cell c){
    return this->copy(c);
  }
  
  std::stringstream to_string (std::string str="") const {
    std::stringstream out;
    out << (str!="" ? "" : str+" ") << this->val;
    return out;
  }

  friend std::ostream& operator<<(std::ostream &out, const Cell& data) {
    out << data.to_string().str();
    return out;
  }
};

#define DISC 4
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

__global__ void kernel(Cell* matrix, int i, Configuration<double>* conf){
  int tidx=threadIdx.x+blockDim.x*blockIdx.x;
  int id=i+tidx*DIM;
  printf("matrix[%d]: %d matrix[%d]: %d\n", id, matrix[id], (tidx*DIM+i+1), matrix[tidx*DIM+i+1]);
  printf("conf[%d]: (%f, %f) conf[%d]: (%f, %f)\n", i, conf[i].x, conf[i].y, tidx, conf[tidx].x, conf[tidx].y);
  matrix[id].val=matrix[tidx*DIM+i+1].val+conf[i].dist(conf[tidx]);
}

int main(){
  cudaError_t cudaErr=cudaSuccess;

  Cell* matrix; 
  Configuration<double>* conf;
  
  cudaMallocManaged(&matrix, SIZE*sizeof(Cell));
  cudaMallocManaged(&conf, DIM*sizeof(Configuration<double>));

  for (int i=DIM; i>0; i--){
    conf[i-1].x=i;
    conf[i-1].y=i;
    conf[i-1].th=0;
  }

  for (int i=0; i<DIM; i++){
    for (int j=0; j<DISC; j++){
      matrix[i*DISC+j].val=i*DISC+j;
    }
  }
  printVM(matrix, DISC, DIM)

  for (int i=0; i<DIM; i++){
    for (int j=0; j<DISC; j++){
      matrix[i*DISC+j].val=0;
    }
  }
  
  for (int i=DIM-1; i>0; i--){
    kernel<<<1, DISC>>>(matrix, i, conf);
    cudaCheckError(cudaGetLastError());
    cudaErr=cudaDeviceSynchronize();  
    cudaCheckError(cudaErr);
    printVM(matrix, DISC, DIM)
    cout << endl;
  }

  cudaFree(matrix);
  return 0;
}
