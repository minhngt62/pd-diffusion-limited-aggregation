#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* 
 * Initialization for LDA
 */
#define n_iters 1500 // Number of iterations
#define grid_size 128 // Size of block
#define tol 0.001 // Tolerance coefficient
#define omega 1.9 // Determine strength of the mixing
#define theta 1 // Determine shape of the object

/* ------------------- */
int i, j, k, r; // Loop variables
int world_rank, world_size, n_rows; // n_rows: no. of rows per processor
int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};

float *C, *Cprev, *alpha, *all_alpha; // Concentration matries & Alpha for tolerance
int *O, *Obelow, *Cd; // Object & candidate matries
float *Cabove, *Cbelow; // First/last rows passed in communication
float *nutris, *nutri; // Nutri of candidates across processors

/* ------------------- */
char fpath[40];
FILE *f;
MPI_Status status;

float max(float a, float b) {
    return a > b ? a : b;
}

/* 
 *  For each growth candidate a random number 
 *  between zero and one is drawn and if
 *  the random number is smaller than the growth probability.
 */
float threshold() {
    return (float) rand() / (float) RAND_MAX;
}

/* ------------------- */
void diffuse() {
    do {
        *alpha = 0;
        
        // red-black ordering (r=0: red cell; o.w)
        for (r = 0; r < 2; ++r) {
            /* 
             *  Set all elements of boundary strips 
             *  to default for receiving new strips latter
             */
            for (i = 0; i < grid_size; ++i) {
                *(Cabove + i) = 0;
                *(Cbelow + i) = 0; 
            }

            /* 
             *  Exchange boundary scripts
             */
            // Send last row to the next processor
            if (world_rank != world_size - 1)
                MPI_Send(C + (n_rows - 1) * grid_size, grid_size, MPI_FLOAT, world_rank + 1, 1, MPI_COMM_WORLD);

            // Send first row to the former processor
            if (world_rank != 0)
                MPI_Send(C, grid_size, MPI_FLOAT, world_rank - 1, 0, MPI_COMM_WORLD);

            // Receive first row from the next processor to Cbelow
            if (world_rank != world_size - 1)
                MPI_Recv(Cbelow, grid_size, MPI_FLOAT, world_rank + 1, 0, MPI_COMM_WORLD, & status);

            // Receive last row from the former processor to Cabove
            if (world_rank != 0)
                MPI_Recv(Cabove, grid_size, MPI_FLOAT, world_rank - 1, 1, MPI_COMM_WORLD, & status);
            
            /* 
             *  Update concentration by diffusion eq.
             */
            for (i = 0; i < n_rows; i++) {
                // First row of the grid
                if (world_rank == 0 && i == 0) 
                    continue;

                // Last row of the grid
                if (world_rank == world_size - 1 && i == n_rows - 1) 
                    continue;
                
                for (j = 0; j < grid_size; ++j) {
                    // If object
                    if (*(O + i * grid_size + j) == 1)
                        continue;
                    // If the cell has different color
                    if ((world_rank * grid_size + i + j) % 2 == r)
                        continue;

                    // Top left corner
                    if (i == 0 && j == 0)
                        *(C + i * grid_size + j) = (omega / 4) * ( * (Cabove + j) + * (C + i * grid_size + grid_size - 1) + * (C + i * grid_size + j + 1) + * (C + (i + 1) * grid_size + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // Top right corner
                    else  if (i == 0 && j == grid_size - 1) 
                        *(C + i * grid_size + j) = (omega / 4) * ( * (Cabove + j) + * (C + i * grid_size + j - 1) + * (C + i * grid_size + 0) + * (C + (i + 1) * grid_size + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // Bottom left corner
                    else if (i == n_rows - 1 && j == 0) // 
                        *(C + i * grid_size + j) = (omega / 4) * ( * (C + (i - 1) * grid_size + j) + * (C + i * grid_size + grid_size - 1) + * (C + i * grid_size + j + 1) + * (Cbelow + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // Bottom right corner
                    else if (i == n_rows - 1 && j == grid_size - 1) 
                        *(C + i * grid_size + j) = (omega / 4) * ( * (C + (i - 1) * grid_size + j) + * (C + i * grid_size + j - 1) + * (C + i * grid_size + 0) + * (Cbelow + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // First column
                    else if (i != 0 && i != n_rows - 1 && j == 0)
                        *(C + i * grid_size + j) = (omega / 4) * ( * (C + (i - 1) * grid_size + j) + * (C + i * grid_size + grid_size - 1) + * (C + i * grid_size + j + 1) + * (C + (i + 1) * grid_size + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // Last column
                    else if (i != 0 && i != n_rows - 1 && j == grid_size - 1) 
                        *(C + i * grid_size + j) = (omega / 4) * ( * (C + (i - 1) * grid_size + j) + * (C + i * grid_size + j - 1) + * (C + i * grid_size + 0) + * (C + (i + 1) * grid_size + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // First row
                    else if (i == 0 && j != 0 && j != grid_size - 1)
                        *(C + i * grid_size + j) = (omega / 4) * ( * (Cabove + j) + * (C + i * grid_size + j - 1) + * (C + i * grid_size + j + 1) + * (C + (i + 1) * grid_size + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // Last row
                    else if (i == n_rows - 1 && j != 0 && j != grid_size - 1)
                        *(C + i * grid_size + j) = (omega / 4) * ( * (C + (i - 1) * grid_size + j) + * (C + i * grid_size + j - 1) + * (C + i * grid_size + j + 1) + * (Cbelow + j)) + (1 - omega) * * (C + i * grid_size + j);
                    // General case: within the boundary
                    else
                        *(C + i * grid_size + j) = (omega / 4) * ( * (C + (i - 1) * grid_size + j) + * (C + i * grid_size + j - 1) + * (C + i * grid_size + j + 1) + * (C + (i + 1) * grid_size + j)) + (1 - omega) * * (C + i * grid_size + j);
                    
                    // Obtain local max alpha
                    *alpha = max(*alpha, fabs(*(C + i * grid_size + j) - *(Cprev + i * grid_size + j)));
                }
            }
        }

        // Gather alpha in all_alpha 
        MPI_Allgather(alpha, 1, MPI_FLOAT, all_alpha, 1, MPI_FLOAT, MPI_COMM_WORLD);
        *alpha = 0;

        // Retrieve global maximal alpha
        for (i = 0; i < world_size; ++i)
            *alpha = max( *alpha, *(all_alpha + i));

        // Update cell previous values with new diffused values
        for (i = 0; i < n_rows; ++i)
            for (j = 0; j < grid_size; ++j)
                * (Cprev + i * grid_size + j) = * (C + i * grid_size + j);

    } while (
        *alpha > tol
    );

    return;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    MPI_Init( & argc, & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, & world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, & world_size);

    printf("Start process %d\n", world_rank);
    
    n_rows = grid_size / world_size;
    snprintf(fpath, 40, "output/parallel/%d.txt", world_rank);
    f = fopen(fpath, "w");
    printf("Process %d, output path: %s\n", world_rank, fpath);

    C = (float * ) malloc(n_rows * grid_size * sizeof(float));
    Cprev = (float * ) malloc(n_rows * grid_size * sizeof(float));
    alpha = (float * ) malloc(sizeof(float));
    all_alpha = (float * ) malloc(world_size * sizeof(float));

    Cabove = (float * ) malloc(grid_size * sizeof(float));
    Cbelow = (float * ) malloc(grid_size * sizeof(float));

    O = (int * ) malloc(n_rows * grid_size * sizeof(int));
    Cd = (int * ) malloc(n_rows * grid_size * sizeof(int));
    Obelow = (int * ) malloc(grid_size * sizeof(int));

    nutris = (float * ) malloc(world_size * sizeof(float));
    nutri = (float * ) malloc(sizeof(float));

    // Cell Initialization
    for (i = 0; i < n_rows; ++i)
        for (j = 0; j < grid_size; ++j) {
            if (world_rank == 0 && i == 0)
                *(C + i * grid_size + j) = 1;
            else
                *(C + i * grid_size + j) = 0;
            *(Cprev + i * grid_size + j) = *(C + i * grid_size + j);
            *(O + i * grid_size + j) = 0;
        }

    // Object initialization
    if (world_rank == world_size - 1)
        *(O + (n_rows - 1) * grid_size + grid_size / 2) = 1;

    /*
     * DLA process
     */
    for (k = 0; k < n_iters; ++k) {
        diffuse();

        // Grow object
        for (i = 0; i < grid_size; ++i)
            *(Obelow + i) = 0;

        /*
         * Exchange boundary strips of the object matrix
         */
        // Send the first row of object to above
        if (world_rank != 0)
            MPI_Send(O, grid_size, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);

        // Receive object from below
        if (world_rank != world_size - 1)
            MPI_Recv(Obelow, grid_size, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, & status);
    
        /*
         * Grow candidate
         */
        // Initialize candidates
        for (i = 0; i < n_rows; ++i)
            for (j = 0; j < grid_size; ++j)
                * (Cd + i * grid_size + j) = 0;
        
        // Get candidates
        * nutri = 0;
        for (i = 0; i < n_rows; ++i) {
            for (j = 0; j < grid_size; ++j) {

                // If object cell
                if ( * (O + i * grid_size + j) == 1)
                    continue;

                // Sum
                int sum = 0;

                // cell (u, v): adjacent cell of cell (i, j)
                for (r = 0; r < 4; ++r) {
                    int u, v;
                    u = i + dx[r];
                    v = j + dy[r];
                    if (u >= 0 && u < n_rows && v >= 0 && v < grid_size && * (O + u * grid_size + v) == 1)
                        sum += 1;
                }

                // If last row, check object below
                if (i == n_rows - 1 && * (Obelow + j) == 1)
                    sum += 1;

                // If sum postive: there is at least 1 adjacent object
                if (sum > 0) {
                    *nutri += pow(*(C + i * grid_size + j), theta);
                    *(Cd + i * grid_size + j) = 1;
                }
            }
        }
        // Gather nutrition
        MPI_Allgather(nutri, 1, MPI_FLOAT, nutris, 1, MPI_FLOAT, MPI_COMM_WORLD);

        // Extract global total nutrition
        float total_nutri = 0.0;
        for (i = 0; i < world_size; ++i)
            total_nutri += * (nutris + i);

        // Grow candidates
        for (i = 0; i < n_rows; ++i)
            for (j = 0; j < grid_size; ++j)
                if (*(Cd + i * grid_size + j) == 1 && threshold() <= (pow(*(C + i * grid_size + j), theta) / total_nutri)) {
                    *(O + i * grid_size + j) = 1;
                    *(C + i * grid_size + j) = 0;
                } 
    }

    // Write the result of each process to an output file
    printf("Process %d, exporting output to: %s\n", world_rank, fpath);
    for (i = 0; i < n_rows; ++i) {
        for (j = 0; j < grid_size; ++j) {
            if (*(O + i * grid_size + j) == 1)
                *(C + i * grid_size + j) = 1;
            fprintf(f, "%lf\t", *(C + i * grid_size + j));
        }
        fprintf(f, "\n");
    }

    MPI_Finalize();
    return 0;
}