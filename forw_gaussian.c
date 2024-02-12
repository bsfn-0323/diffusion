//By Stefano Bae
//Forward Ornstein-Uhlenbeck process for training
//This program generates an initial gaussian configuration of P walkers of dimension N
//and evolves them with an OU process at temperature T
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <time.h>
#define PI acos(-1.0)
#define DIAG
#define CUSTOM_DATA
#define DATA_SIZE 1000000
double norm(){
    double r1  = (double)rand()/RAND_MAX;
    double r2  = (double)rand()/RAND_MAX;
    return sqrt(-2.*log(r1))* cos(2.*M_PI*r2);
}

double rand01(){
    return (double)rand()/RAND_MAX;
}

void get_corr_mat(int M, int N, gsl_matrix *corr, double **mat){
    double sumxixj = 0.;
    double sumxi = 0.,sumxj = 0.;
    for(int i = 0;i<N;i++){
        for(int j = i;j<N;j++){
            sumxixj = 0.;sumxi = 0.;sumxj = 0.;
            for(int ex = 0;ex<M;ex++){
                sumxixj += mat[i][ex]*mat[j][ex];
                sumxi += mat[i][ex];
                sumxj += mat[j][ex];
            }
            gsl_matrix_set(corr,i,j,(double)(sumxixj/M - sumxi/M * sumxj/M));
            gsl_matrix_set(corr,j,i,gsl_matrix_get(corr,i,j));
        }
    }
}
void get_mean(int M, int N, gsl_vector *mean, double **mat){
    double sumxi = 0.;
    for(int i = 0;i<N;i++){
        sumxi=0.;
        for(int ex = 0;ex<M;ex++){
        sumxi += mat[i][ex];
        }
        gsl_vector_set(mean,i,(double)sumxi/M);
     // printf("%.4f\t",gsl_vector_get(mean,i));
    }
    
}
gsl_matrix *matsum(const gsl_matrix *A,const gsl_matrix *B,int dim){
    gsl_matrix *result = gsl_matrix_alloc(dim,dim);
    for(int i= 0;i<dim;i++){
        for(int j= 0;j<dim;j++){
            gsl_matrix_set(result,i,j,gsl_matrix_get(A,i,j)+gsl_matrix_get(B,i,j));
        }
    }
    return result;
}
gsl_matrix *matscale(const gsl_matrix *A,double c,int dim){
    gsl_matrix *result = gsl_matrix_alloc(dim,dim);
    for(int i= 0;i<dim;i++){
        for(int j= 0;j<dim;j++){
            gsl_matrix_set(result,i,j,gsl_matrix_get(A,i,j)*c);
        }
    }
    return result;
}
gsl_vector *vecscale(const gsl_vector *A,double c,int dim){
    gsl_vector *result = gsl_vector_alloc(dim);
    for(int i= 0;i<dim;i++){
        gsl_vector_set(result,i,gsl_vector_get(A,i)*c);
    }
    return result;
}
void print_matrix(gsl_matrix *mat,int dim,char *fname){
    FILE *fp;
    fp = fopen(fname,"w");
    for(int i = 0;i<dim;i++){
        for(int j = 0;j<dim;j++){
            fprintf(fp,"%.10g ",gsl_matrix_get(mat,i,j));
        }
        fprintf(fp,"\n");
    }
}
void importBinaryToArray2(const char *filename, int **array, int dim1, int dim2){
    FILE *file = fopen(filename, "rb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file for reading.\n");
    }
    for (int i = 0; i < dim1; i++) {
        size_t read = fread(array[i], sizeof(int), dim2, file);
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
void exportArrayToBinary3(const char *filename, double ***array, int dim1, int dim2, int dim3) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        return;
    }

    // Write the array data to the file
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            fwrite(array[i][j], sizeof(double), dim3, file);
        }
    }

    fclose(file);
}
void free_array3(double ***arr,int dim1,int dim2){
    for(int i = 0;i<dim1;i++){
        for(int j= 0;j<dim2;j++){
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}
void free_array2(double **arr,int dim){
    for(int i = 0;i<dim;i++){
        free(arr[i]);
    }
    free(arr);
}
gsl_matrix *inv_mat(int dim,gsl_matrix* mat){
    gsl_permutation *p = gsl_permutation_alloc(dim);
    int s;
    gsl_linalg_LU_decomp(mat,p,&s);
    gsl_matrix *inv = gsl_matrix_alloc(dim,dim);
    gsl_linalg_LU_invert(mat,p,inv);
    gsl_permutation_free(p);
    return inv;
}
int main(int argc, char **argv){
    srand((unsigned int)time(NULL));

    int N,P;    //N = internal dimension, P = number of samples
    int nSteps;
    double dt;
    double T;   //temperature
    char filename[200],dir[200];
    if (argc != 7) {
        fprintf(stderr, "usage: %s <N> <P> <T> <nSteps> <dt>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    N = atoi(argv[1]);
    P = atoi(argv[2]);
    T = atof(argv[3]);
    nSteps = atoi(argv[4]);
    dt = atof(argv[5]);
    strcpy(dir, argv[6]);
    #ifndef DIAG
    gsl_matrix *cov= gsl_matrix_alloc(N,N);
    gsl_matrix *lowtri= gsl_matrix_alloc(N,N);
    gsl_matrix *aux= gsl_matrix_alloc(N,N);

    //Build covariance matrix, and Cholesky decomposition
    double sum = 0.;
    //First build a positive defined matrix
    for(int i = 0;i<N;i++){
        for(int j = 0;j<=i;j++){
            gsl_matrix_set(aux,i,j,norm());
            gsl_matrix_set(aux,j,i,gsl_matrix_get(aux,i,j));
        }
    }
    for(int i = 0;i<N;i++){
        double tmp;
        for(int j = 0;j<N;j++){
            tmp = 0.;
            for(int k = 0;k<N;k++){
                tmp += gsl_matrix_get(aux,i,k)*gsl_matrix_get(aux,j,k);
            }
            gsl_matrix_set(cov,i,j,tmp);
        }
    }
    //Then get the low triangular matrix from Cholesky decomp
    sprintf(filename,"cov_N%d_P%d_T%.2f_dt%.2f.txt",N,P,T,dt);
    print_matrix(cov,N,filename);
    gsl_linalg_cholesky_decomp(cov);
    gsl_matrix_memcpy(lowtri,cov);
    print_matrix(lowtri,N,"ltr_mat.txt");
    #endif

    //Initialize walkers
    double*** walkers = (double***)malloc((nSteps+1)*sizeof(double**));
    double **v = (double**)malloc(N*sizeof(double*));

    for(int i = 0;i<nSteps+1;i++){
        walkers[i] = (double**)malloc(N*sizeof(double*));
        for(int j = 0;j<N;j++){
            walkers[i][j] = (double*)malloc(P*sizeof(double));
        }
    }
    #ifndef CUSTOM_DATA
    for(int i=0;i<N;i++){
        v[i] = (double*)malloc(P*sizeof(double));
        for(int j =0;j<P;j++){
            v[i][j] = norm();
        }
    }

    #ifndef DIAG
    for(int ex =0;ex<P;ex++){
        for(int i = 0;i<N;i++){
            sum = (double)i;
            for(int j = 0;j<=i;j++){
                sum+= gsl_matrix_get(lowtri,i,j)*v[j][ex];
            }
            walkers[0][i][ex] = sum;
            
        }
    }
    #endif
    #ifdef DIAG                

    for(int ex =0;ex<P;ex++){
        for(int i = 0;i<N;i++){
            walkers[0][i][ex] = 1.+v[i][ex];
        }
    }
    #endif
    #endif
    #ifdef CUSTOM_DATA
    int **x0 = (int**)malloc(DATA_SIZE*sizeof(int*));
    for(int i = 0;i<DATA_SIZE;i++)
        x0[i] = (int*)malloc(N*sizeof(int));
    importBinaryToArray2(dir,x0,P,N);
    for(int ex =0;ex<P;ex++){
        for(int i = 0;i<N;i++){
            walkers[0][i][ex] = (double)x0[ex][i];
        }
    }
    #endif
    //Evolve the walkers with OU process
    /*for(int i = 0;i<N;i++){
        for(int j = 0;j<P;j++){
            for(int k = 1;k<nSteps+1;k++){
                walkers[k][i][j] = walkers[k-1][i][j]*(1.-dt) + sqrt(2.*T*dt)*norm();
            }
        }
    }
    sprintf(filename,"ising_diffusion/forwOU_N%d_P%d_T%.2f_dt%.2f.bin",N,P,T,dt);
    exportArrayToBinary3(filename, walkers, nSteps+1,N, P);*/
    //sprintf(filename,"init_forwOU_N%d_P%d_T%.2f_dt%.2f.bin",N,P,T,dt);
    //exportArrayToBinary2(filename,walkers[0],N,P);
    printf("Computing Score\n");
    //Score function
    //Definition of weights and offsets
    double*** W = (double***)malloc((nSteps+1)*sizeof(double**));
    double ** b = (double**)malloc((nSteps+1)*sizeof(double*));
    gsl_matrix *Wt = gsl_matrix_alloc(N,N);
    gsl_matrix *C0 = gsl_matrix_alloc(N,N);
    gsl_matrix *Id = gsl_matrix_alloc(N,N);
    gsl_vector *bt = gsl_vector_alloc(N);
    gsl_vector *m0 = gsl_vector_alloc(N);
    for(int i = 0;i<nSteps+1;i++){
        W[i] = (double**)malloc(N*sizeof(double*));
        b[i] = (double*)malloc(N*sizeof(double));
        for(int j = 0;j<N;j++){
            W[i][j] = (double*)malloc(N*sizeof(double));
        }
    }
    //Get initial correlation matrix C0
    get_corr_mat(P,N,C0,walkers[0]);
    get_mean(P,N,m0,walkers[0]);
    gsl_matrix_set_identity(Id);
    //print_matrix(C0,N,"init_cov_mat.txt");
    //Build W matrix
    for(int t = 0;t<=nSteps ;t++){
        Wt = inv_mat(N,matsum( matscale(Id,T*(1.-exp(-2.*t*dt)),N), matscale(C0,exp(-2.*t*dt),N),N));
        gsl_blas_dgemv(CblasNoTrans ,-1.0*exp(-t*dt), Wt,m0,0.0,bt);
        for(int i = 0;i<N;i++){
            for(int j = i;j<N;j++){
                W[t][i][j]=gsl_matrix_get(Wt,i,j);
                W[t][j][i]=gsl_matrix_get(Wt,j,i);
            }
            b[t][i]=gsl_vector_get(bt,i);
        }
    }

    sprintf(dir,"mkdir score_N%d_T%.3f_P%d",N,T,P);
    system(dir);
    sprintf(filename,"score_N%d_T%.3f_P%d/W.bin",N,T,P);
    exportArrayToBinary3(filename, W, nSteps+1,N, N);
    sprintf(filename,"score_N%d_T%.3f_P%d/b.bin",N,T,P);
    exportArrayToBinary2(filename, b, nSteps+1,N);
    FILE *fp;
    sprintf(filename,"score_N%d_T%.3f_P%d/settings.txt",N,T,P);
    fp = fopen(filename,"w");
    fprintf(fp,"#nSteps\tdt\n");
    fprintf(fp,"%d\t%f\n",nSteps,dt);
    fclose(fp);
    #ifndef DIAG
    gsl_matrix_free(cov);
    gsl_matrix_free(aux);
    gsl_matrix_free(lowtri);
    #endif
    gsl_matrix_free(Wt);
    gsl_matrix_free(C0);
    gsl_matrix_free(Id);
    gsl_vector_free(bt);
    gsl_vector_free(m0);
    free_array2(v,N);
    free_array2(b,nSteps+1);
    free_array3(walkers,nSteps+1,N);
    free_array3(W,nSteps+1,N);

    return 0;
}
