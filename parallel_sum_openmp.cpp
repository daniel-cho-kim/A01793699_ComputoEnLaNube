#include <omp.h>
#include <iostream>

static const int N = 200;
int main(){
    int is_cpu = true;
    int *arrayA = static_cast<int*>(malloc(N * sizeof(int)));
    for(int i=0; i<N; i++) arrayA[i] = i;

    #pragma omp target map(from:is_cpu) map(tofrom:arrayA[0:N])
    {
        is_cpu=omp_is_initial_device();
        #pragma omp parallel for
        for (int i=0; i<N; i++) 
            arrayA[i]=rand()%100;
    }
    printf ("Running ArrayA on %s\n", (is_cpu?"CPU":"GPU"));
    for(int i=0; i<N; i++) std::cout << arrayA[i] << std::endl;
 


    int *arrayB = static_cast<int*>(malloc((N+1) * sizeof(int)));
    for(int i=0; i<N; i++) arrayB[i] = i;

    #pragma omp target map(from:is_cpu) map(tofrom:arrayB[0:N])
    {
        is_cpu=omp_is_initial_device();
        #pragma omp parallel for
        for (int i=0; i<N; i++) 
            arrayB[i]=rand()%100;
    }
    printf ("Running ArrayB on %s\n", (is_cpu?"CPU":"GPU"));
    for(int i=0; i<N; i++) std::cout << arrayB[i] << std::endl;
    


    int *sum = static_cast<int*>(malloc((N+2) * sizeof(int)));
    for(int i=0; i<N; i++) sum[i] = i;

    #pragma omp target map(from:is_cpu) map(tofrom:sum[0:N])
    {
        is_cpu=omp_is_initial_device();
        #pragma omp parallel for
        for (int i=0; i<N; i++) 
            sum[i]=arrayA[i]+arrayB[i];
    }
    printf ("Running SUM on %s\n", (is_cpu?"CPU":"GPU"));
    for(int i=0; i<N; i++) std::cout << sum[i] << std::endl;
    
    free(arrayA);
    free(arrayB);
    free(sum);
    return 0;
}