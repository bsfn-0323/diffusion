//By Stefano Bae
//Backward Ornstein-Uhlenbeck process for generating
//This program generates an initial gaussian configuration of P walkers of dimension N
//and evolves them with a backward OU process
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <time.h>
#define PI acos(-1.0)
int N,P;    //N = internal dimension, P = number of samples
double norm(){
    double r1  = (double)rand()/RAND_MAX;
    double r2  = (double)rand()/RAND_MAX;
    return sqrt(-2.*log(r1))* cos(2.*M_PI*r2);
}
void exportArrayToBinary3(const char *filename, double ***array, int dim1, int dim2, int dim3) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
    }

    // Write the array data to the file
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            fwrite(array[i][j], sizeof(double), dim3, file);
        }
    }

    fclose(file);
}
void exportArrayToBinary2(const char *filename, double **array, int dim1, int dim2) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
    }

    // Write the array data to the file
    for (int i = 0; i < dim1; i++) {
        fwrite(array[i], sizeof(double), dim2, file);
    }

    fclose(file);
}

void importBinaryToArray3(const char *filename, double ***array, int dim1, int dim2, int dim3){
    FILE *file = fopen(filename, "rb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading.\n");
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            size_t read =fread(array[i][j], sizeof(double), dim3, file);
        }
    }
    fclose(file);
    
}
void importBinaryToArray2(const char *filename, double **array, int dim1, int dim2){
    FILE *file = fopen(filename, "rb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading.\n");
    }
    for (int i = 0; i < dim1; i++) {
        size_t read = fread(array[i], sizeof(double), dim2, file);
    }
    fclose(file);
}
void free_3d_array(double*** array, int size1, int size2) {
	    // Iterate over the first dimension
	    for (int i = 0; i < size1; i++) {
	        // Iterate over the second dimension
	        for (int j = 0; j < size2; j++) {
	            // Free the memory allocated for the innermost dimension
	            free(array[i][j]);
	        }
	        // Free the memory allocated for the second dimension
	        free(array[i]);
	    }
	    // Free the memory allocated for the first dimension
	    free(array);
	}
void free_2d_array(double** array, int size1) {
	    // Iterate over the first dimension
	    for (int i = 0; i < size1; i++) {
	        // Free the memory allocated for the second dimension
	        free(array[i]);
	    }
	    // Free the memory allocated for the first dimension
	    free(array);
	}
double delta(int i,int j){
    return (double) (i==j);
}

void scoreEval(double ***weights,double **offsets,double ***walkers, double *score,int i,int j, int t){
    double s = - offsets[t][i];
        for(int k = 0;k<N;k++){
            s+= -1.0*weights[t][i][k]*walkers[t][k][j];
        }
    (*score) = s;
}

int main(int argc, char **argv){
    //change with something better
    srand((unsigned int)time(NULL));

    
    int nSteps;
    double dt;
    double T;   //temperature
    char filename[200];
    if (argc != 6) {
        fprintf(stderr, "usage: %s <N> <P> <T> <nSteps> <dt>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    N = atoi(argv[1]);
    P = atoi(argv[2]);
    T = atof(argv[3]);
    nSteps = atoi(argv[4]);
    dt = atof(argv[5]);
    
    double*** walkers = (double***)malloc((nSteps+1)*sizeof(double**));
    double*** weights = (double***)malloc((nSteps+1)*sizeof(double**));
    double** offsets = (double**)malloc((nSteps+1)*sizeof(double*));

    for(int i = 0;i<nSteps+1;i++){
        walkers[i] = (double**)malloc(N*sizeof(double*));
        weights[i] = (double**)malloc(N*sizeof(double*));
        offsets[i] = (double*)malloc(N*sizeof(double));
        for(int j = 0;j<N;j++){
            walkers[i][j] = (double*)malloc(P*sizeof(double));
            weights[i][j] = (double*)malloc(N*sizeof(double));
        }
    }
    
    //Import weights and offsets
    sprintf(filename,"score_N%d_T%.3f_P%d/W.bin",N,T,P);
    importBinaryToArray3(filename,weights,nSteps+1,N,N);
    sprintf(filename,"score_N%d_T%.3f_P%d/b.bin",N,T,P);
    importBinaryToArray2(filename,offsets,nSteps+1,N);
    //importBinaryToArray3(filename,walkers,nSteps+1,N,P);
    //Backward
    double score;
    for(int i = 0;i<N;i++){
        for(int j = 0;j<P;j++){
           walkers[nSteps][i][j] = sqrt(T)*norm();
            for(int t = nSteps; t>0;t--){
                scoreEval(weights,offsets,walkers,&score,i,j,t);  
                walkers[t-1][i][j] = walkers[t][i][j]*(1.+dt) - 2.*T*score*dt + sqrt(2.*T*dt)*norm();
                //walkers[t-1][i][j]= (walkers[t][i][j] + 2.*T*score*dt + sqrt(2.*T*dt)*norm())/(1.-dt);
                //walkers[t-1][i][j] = 0.5*(y1+y2);
            }
        }
    }

    //sprintf(filename,"mkdir score_N%d_T%.3f_P%d",N,T,P);
    sprintf(filename,"score_N%d_T%.3f_P%d/backOU.bin",N,T,P);
    exportArrayToBinary3(filename, walkers, nSteps+1, N, P);
    sprintf(filename,"score_N%d_T%.3f_P%d/init_backOU.bin",N,T,P);
    exportArrayToBinary2(filename,walkers[0],N,P);
    
    free_3d_array(walkers,nSteps+1,N);
    free_3d_array(weights,nSteps+1,N);
    free_2d_array(offsets,nSteps+1);
}
