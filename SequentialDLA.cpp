#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define iters 800	// Number of iterations	
#define N 128		// Size of block
#define tol 0.001	// Tolerance coefficient 
#define omega 1.9	// Determine strength of the mixing
#define theta 1		// Determine shape of the object

int *object, *candidates;
float *C, *C_prev;

float max(float a, float b) {
    return a>b ? a : b;
}

float uniform(){
    return (float)rand() / (float)RAND_MAX;
}

void jacobi_iterative() {
	float diff;
    do {
        diff = 0;
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < N; ++j){
                // Pass if the cell is filled with object
                if(*(object + i*N + j) == 1)
                    continue;
				
				// Handle first row 0
                if(i == 0)
                    continue;

                // Handle last row (size - 1)
                if(i == N - 1)
                    continue;
                    
                // Handle first cell of each process
                if(i != 0 && i != N - 1 && j == 0){
                    // Take last column for the left neighborhood.
					*(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + N-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                }else {
                    if(i !=0 && i != N - 1 && j == N-1){
                        *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + 0) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                    }else {
                        // general case
                        *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                    }
                }
                diff = max(diff, fabs(*(C + i*N + j) - *(C_prev + i*N + j)));
                }
        }
        // Update the last cell's C
        for(int i=0; i < N; ++i)
            for(int j=0; j < N; ++j)
                *(C_prev + i*N + j) = *(C + i*N + j);

    } while(diff > tol);

    return;
}

int main(int argc, char *argv[])
{	
	float alpha, exp_total_C;
	
	clock_t start, end;
	int i, j, r;
	int dx[4] = {-1,0,0,1}, dy[4] = {0,-1,1,0};
	

	char snum[40];
	char fpath[40] = "./";
	FILE *fp;

    start = clock();
    srand(time(NULL));

    strcat(snum, "output/sequential/out.txt");
    strcat(fpath, snum);
    fp = fopen(fpath, "w");

    C = (float *) malloc(N * N * sizeof(float)); //
    C_prev = (float *) malloc(N * N * sizeof(float));


    object = (int *) malloc(N * N * sizeof(int));
    candidates = (int *) malloc(N * N * sizeof(int));

    // C initialization
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j){
            if (i == 0)
                *(C + i*N + j) = 1;
            else
                *(C + i*N + j) = 0;

            *(C_prev + i*N + j) = *(C + i*N + j);
            *(object + i*N + j) = 0;
        }

    // Object initialization
    *(object + (N - 1)*N + N/2) = 1;

    for(int iter = 0; iter < iters; ++iter) {
        // Solving Laplace equation using Jacobi iterative method
        jacobi_iterative();

		// Grow
        for(i = 0; i < N; ++i)
            for(j = 0; j < N; ++j)
                *(candidates + i*N + j) = 0;

        exp_total_C = 0;

        //Determine the cell is able to grow
        for(i = 0; i < N; ++i)
            for(j = 0; j < N; ++j){
                if(*(object + i*N + j) == 1)
                    continue;

                int sum = 0;

                for(r = 0; r < 4; ++r){
                    int u, v;
                    u = i + dx[r];
                    v = j + dy[r];
                    if (u>=0 && u<N && v>=0 && v<N && *(object + u*N + v) == 1)
                        sum += 1;
                }

                //If the cell is able to grow, increase total concentration and mark the cell as a candidate
                if(sum > 0){
                    exp_total_C += pow(*(C + i*N + j), theta);
                    *(candidates + i*N + j) = 1;
                }
            }


        //Randomly choose the cell to grow among candidates
        for(i = 0; i < N; ++i)
            for(j = 0; j < N; ++j)
                if(*(candidates + i*N + j) == 1 && uniform() <= (pow(*(C + i*N +j), theta)/(exp_total_C))){
                    *(object + i*N + j) = 1;
                    *(C + i*N + j) = 0;
                }

        fprintf(fp, "Iter %d:\n", iter);
        end = clock();
        fprintf(fp, "%.3f\n", ((double)(end - start) / CLOCKS_PER_SEC));
        for (i = 0; i < N; ++i){
            for (j = 0; j < N; ++j){
                float temp = *(C + i*N + j);
                if(*(object + i*N + j) == 1)
                    temp = 1;
                fprintf(fp, "%lf\t", temp);
            }
            fprintf(fp, "\n");
        }
    }

    //Finally, update the C and write into file
    jacobi_iterative();

    fprintf(fp, "Output:\n");
    for(i = 0; i < N; ++i){
        for(j = 0; j < N; ++j){
            if(*(object + i*N + j) == 1)
                *(C + i*N + j) = 1;
            fprintf(fp, "%lf\t", *(C + i*N + j));
        }
        fprintf(fp, "\n");
    }

	return 0;
}
