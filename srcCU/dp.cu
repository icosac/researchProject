#ifdef CUDA_ON
#include <dp.cuh>

void DP::guessInitialAngles(std::vector<Configuration2<double> >& points, const std::vector<bool> fixedAngles){
  for (uint i=0; i<points.size()-1; i++){
    if (!fixedAngles[i]){    	
      double diffX=points[i].x()-points[i+1].x();
      double diffY=points[i].y()-points[i+1].y();
      points[i].th(atan2(diffY, diffX));
    }
  }
}

__global__ void printResults(real_type* results, size_t discr, size_t size){
  for (int i=0; i<size; i++){
    for(int j=0; j<discr; j++){
      for(int h=0; h<discr; h++){
        printf("(%2.0f,%.2f)", (float)((i*discr+j)*discr+h), results[(i*discr+j)*discr+h]);
      }
      printf("\t");
    }
    printf("\n");
  }
}

#endif 

