#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "bb_segsort.h"
//#include <cub/cub.cuh>

#define ERR fprintf(stderr, "ERR\n");

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FTYPE float
#define STYPE int

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-2147483648)
#define THRESHOLD (8*2*1)
#define BH (128/1)
#define BW (64/1)
#define MIN_OCC (BW*3/4)
//#define MIN_OCC (BW/4)
//#define BW (
#define SBSIZE (1024/8)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define SC_SIZE (2048)

//#define SIM_VALUE

#define GPRINT(x,y) int *tt0=(int *)malloc(sizeof(int)*(y));\
	fprintf(stderr, "\n");\
	cudaMemcpy(tt0, x, sizeof(int)*(y), cudaMemcpyDeviceToHost);\
	for(int i=0;i<(y);i++) fprintf(stderr,"%d ", tt0[i]); fprintf(stderr,"\n");\
	free(tt0);

#define GPRINT2(x,y) int *tt1=(int *)malloc(sizeof(int)*(y));\
	fprintf(stderr, "\n");\
	cudaMemcpy(tt1, x, sizeof(int)*(y), cudaMemcpyDeviceToHost);\
	for(int i=0;i<(y);i++) fprintf(stderr,"%d ", tt1[i]); fprintf(stderr,"\n");\
	free(tt1);


int gran=1;

struct v_struct {
	int row, col;
	FTYPE val;
	int grp;
};

double avg, vari;
struct v_struct *temp_v, *gold_temp_v;
int sc, nr, nc, ne, gold_ne, npanel, mne, mne_nr;
int nr0;

int *csr_v; 
int *csr_e0;
FTYPE *csr_ev0;
FTYPE *ocsr_ev;

//int *mcsr_v;
int *mcsr_e; // can be short type
int *mcsr_cnt;
int *mcsr_list;

int *baddr, *saddr;
int num_dense;

int *special;
int *special2;
int special_p;

long datavol;


int compare0(const void *a, const void *b)
{
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}

void ready(int argc, char **argv)
{
        FILE *fp;
        int *loc;
        char buf[300];
        int nflag, sflag;
        int pre_count=0;
        int i;

        srand(time(NULL));

        sc = atoi(argv[2]);
        fp = fopen(argv[1], "r");
        fgets(buf, 300, fp);
        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

#ifdef SYM
        sflag = 1;
#endif

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &nr, &nc, &ne);
        nr0 = nr;
        ne *= (sflag+1);
        nr = CEIL(nr,BH)*BH;
	npanel = CEIL(nr,BH);

        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
        gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));

        for(i=0;i<ne;i++) {
                fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
		temp_v[i].grp = INIT_GRP;
                temp_v[i].row--; temp_v[i].col--;

                if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
                        exit(0);
                }
                if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
                else if(nflag == 1) {
                        FTYPE ftemp;
                        fscanf(fp, " %f ", &ftemp);
                        temp_v[i].val = ftemp;
                } else { // complex
                        FTYPE ftemp1, ftemp2;
                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                        temp_v[i].val = ftemp1;
                }
temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
#ifdef SIM_VALUE
temp_v[i].val = 1.0f;
#endif

                if(sflag == 1) {
                        i++;
                        temp_v[i].row = temp_v[i-1].col;
                        temp_v[i].col = temp_v[i-1].row;
                        temp_v[i].val = temp_v[i-1].val;
        		temp_v[i].grp = INIT_GRP;
	        }
        }
        qsort(temp_v, ne, sizeof(struct v_struct), compare0);

        loc = (int *)malloc(sizeof(int)*(ne+1));

        memset(loc, 0, sizeof(int)*(ne+1));
        loc[0]=1;
        for(i=1;i<ne;i++) {
                if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
                        loc[i] = 0;
                else loc[i] = 1;
        }
        for(i=1;i<=ne;i++)
                loc[i] += loc[i-1];
        for(i=ne; i>=1; i--)
                loc[i] = loc[i-1];
        loc[0] = 0;

        for(i=0;i<ne;i++) {
                temp_v[loc[i]].row = temp_v[i].row;
                temp_v[loc[i]].col = temp_v[i].col;
                temp_v[loc[i]].val = temp_v[i].val;
                temp_v[loc[i]].grp = temp_v[i].grp;
        }
        ne = loc[ne];
        temp_v[ne].row = nr;
        gold_ne = ne;
        for(i=0;i<=ne;i++) {
                gold_temp_v[i].row = temp_v[i].row;
                gold_temp_v[i].col = temp_v[i].col;
                gold_temp_v[i].val = temp_v[i].val;
                gold_temp_v[i].grp = temp_v[i].grp;
        }
        free(loc);

        csr_v = (int *)malloc(sizeof(int)*(nr+1));
        csr_e0 = (int *)malloc(sizeof(int)*ne+256);
        csr_ev0 = (FTYPE *)malloc(sizeof(FTYPE)*ne+256);
        ocsr_ev = (FTYPE *)malloc(sizeof(FTYPE)*ne+256);
        memset(csr_v, 0, sizeof(int)*(nr+1));

        for(i=0;i<ne;i++) {
                csr_e0[i] = temp_v[i].col;
                csr_ev0[i] = temp_v[i].val;
                csr_v[1+temp_v[i].row] = i+1;
        }

        for(i=1;i<nr;i++) {
                if(csr_v[i] == 0) csr_v[i] = csr_v[i-1];
        }
        csr_v[nr] = ne;

        // fprintf(stdout,"TTAAGG,%s,%d,%d,%d,",argv[1],nr0,nc,ne);

}

__global__
__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel128_sparse_v2l(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev)
{
        int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+2))+lane;
        int offset2 = offset + MFACTOR;
        int offset3 = offset + MFACTOR*2;
        int offset4 = offset + MFACTOR*3;
        int i, j;


	//FTYPE r, r2;
	FTYPE o1 = vout[idx*sc+offset];
	FTYPE o2 = vout[idx*sc+offset2];
	FTYPE o3 = vout[idx*sc+offset3];
	FTYPE o4 = vout[idx*sc+offset4];

	int dummy = mcsr_cnt[idx/BH]*BH + ((idx&(BH-1))+1)*(mcsr_cnt[idx/BH+1] - mcsr_cnt[idx/BH]);
	int loc1 = mcsr_e[dummy-1], loc2 = mcsr_e[dummy];

	loc1 += ((loc2 - loc1)/STHRESHOLD)*STHRESHOLD;

        int buf; //FTYPE buf2;
        int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
        int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

	int jj=0, l;
	FTYPE r0=0.0f;

        buf = csr_e[loc1+lane];
	for(l=loc1; l<interm2; l+=4) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		int e2 = __shfl_sync(0xffffffff, buf, jj+1,MFACTOR)*sc;
		int e3 = __shfl_sync(0xffffffff, buf, jj+2,MFACTOR)*sc;
		int e4 = __shfl_sync(0xffffffff, buf, jj+3,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;
		FTYPE r2 = vin[e2+offset] * o1;
		r2 += vin[e2+offset2] * o2;
		r2 += vin[e2+offset3] * o3;
		r2 += vin[e2+offset4] * o4;
		FTYPE r3 = vin[e3+offset] * o1;
		r3 += vin[e3+offset2] * o2;
		r3 += vin[e3+offset3] * o3;
		r3 += vin[e3+offset4] * o4;
		FTYPE r4 = vin[e4+offset] * o1;
		r4 += vin[e4+offset2] * o2;
		r4 += vin[e4+offset3] * o3;
		r4 += vin[e4+offset4] * o4;

		int lv = (lane & 3);
		if(lv == 1) {
			FTYPE t = r2; r2 = r1; r1 = t;
			t = r4; r4 = r3; r3 = t;	
		} else if(lv == 2){
			FTYPE t = r3; r3 = r1; r1 = t;
			t = r4; r4 = r2; r2 = t;	
		} else if(lv == 3){
			FTYPE t = r4; r4 = r1; r1 = t;
			t = r3; r3 = r2; r2 = t;	
		} 

		r2 = __shfl_xor_sync(0xffffffff, r2, 1);
		r3 = __shfl_xor_sync(0xffffffff, r3, 2);
		r4 = __shfl_xor_sync(0xffffffff, r4, 3);
		r1 += r2+r3+r4;
		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);

		if(lane >= jj && lane < jj+4) {
			r0 = r1;
		}
                jj = ((jj+4)&(MFACTOR-1));
                if(jj == 0) {
		        buf = csr_e[l+4+lane];
			//atomicAdd(&ocsr_ev[l-28+lane], r0 * csr_ev[l-28+lane]);
			ocsr_ev[l-28+lane] = r0 * csr_ev[l-28+lane];
                }
	}
	if(interm2 < interm3) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		int e2 = __shfl_sync(0xffffffff, buf, jj+1,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;
		FTYPE r2 = vin[e2+offset] * o1;
		r2 += vin[e2+offset2] * o2;
		r2 += vin[e2+offset3] * o3;
		r2 += vin[e2+offset4] * o4;

		int lv = (lane & 1);
		if(lv == 1) {
			FTYPE t = r2; r2 = r1; r1 = t;
		}
 
		r2 = __shfl_xor_sync(0xffffffff, r2, 1);
		r1 += r2;
		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);
		r1 += __shfl_xor_sync(0xffffffff, r1, 2);

		if(lane >= jj && lane < jj+2) {
			r0 = r1;
		}

                jj = (jj+2);
	}
	if(interm3 < loc2) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;

		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);
		r1 += __shfl_xor_sync(0xffffffff, r1, 2);
		r1 += __shfl_xor_sync(0xffffffff, r1, 1);

		if(lane == jj) {
			r0 = r1;
		}

		jj = (jj+1);
	}
	l = loc1+(((l - loc1)>>5)<<5);
	if(lane < jj) {
		//atomicAdd(&ocsr_ev[l+lane], r0 * csr_ev[l+lane]);
		ocsr_ev[l+lane] = r0 * csr_ev[l+lane];
	}
}	


__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel128_sparse_v2h(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev, int *special, int *special2)
{
        int idx = special[blockIdx.x];// + (threadIdx.x>>(LOG_MFACTOR));
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
        int offset2 = offset + MFACTOR;
        int offset3 = offset + MFACTOR*2;
        int offset4 = offset + MFACTOR*3;
        int i, j;

	FTYPE o1 = vout[idx*sc+offset];
	FTYPE o2 = vout[idx*sc+offset2];
	FTYPE o3 = vout[idx*sc+offset3];
	FTYPE o4 = vout[idx*sc+offset4];

	int dummy = mcsr_cnt[idx/BH]*BH + ((idx&(BH-1))+1)*(mcsr_cnt[idx/BH+1] - mcsr_cnt[idx/BH]);

	int loc1 = mcsr_e[dummy-1] + special2[blockIdx.x] + ((threadIdx.x>>5)*SSTRIDE);

        int buf; //FTYPE buf2;
        //int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
        //int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

	int jj=0, l;
	FTYPE r0=0.0f;

        buf = csr_e[loc1+lane];
	for(l=loc1; l<loc1+SSTRIDE; l+=4) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		int e2 = __shfl_sync(0xffffffff, buf, jj+1,MFACTOR)*sc;
		int e3 = __shfl_sync(0xffffffff, buf, jj+2,MFACTOR)*sc;
		int e4 = __shfl_sync(0xffffffff, buf, jj+3,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;
		FTYPE r2 = vin[e2+offset] * o1;
		r2 += vin[e2+offset2] * o2;
		r2 += vin[e2+offset3] * o3;
		r2 += vin[e2+offset4] * o4;
		FTYPE r3 = vin[e3+offset] * o1;
		r3 += vin[e3+offset2] * o2;
		r3 += vin[e3+offset3] * o3;
		r3 += vin[e3+offset4] * o4;
		FTYPE r4 = vin[e4+offset] * o1;
		r4 += vin[e4+offset2] * o2;
		r4 += vin[e4+offset3] * o3;
		r4 += vin[e4+offset4] * o4;

		int lv = (lane & 3);
		if(lv == 1) {
			FTYPE t = r2; r2 = r1; r1 = t;
			t = r4; r4 = r3; r3 = t;	
		} else if(lv == 2){
			FTYPE t = r3; r3 = r1; r1 = t;
			t = r4; r4 = r2; r2 = t;	
		} else if(lv == 3){
			FTYPE t = r4; r4 = r1; r1 = t;
			t = r3; r3 = r2; r2 = t;	
		} 

		r2 = __shfl_xor_sync(0xffffffff, r2, 1);
		r3 = __shfl_xor_sync(0xffffffff, r3, 2);
		r4 = __shfl_xor_sync(0xffffffff, r4, 3);
		r1 += r2+r3+r4;
		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);

		if(lane >= jj && lane < jj+4) {
			r0 = r1;
		}
                jj = ((jj+4)&(MFACTOR-1));
                if(jj == 0) {
		        buf = csr_e[l+4+lane];
			//atomicAdd(&ocsr_ev[l-28+lane], r0 * csr_ev[l-28+lane]);
			ocsr_ev[l-28+lane] = r0 * csr_ev[l-28+lane];
                }
	}

//if(threadIdx.x == 0) printf("%d %f %f\n", blockIdx.x, r, r2);
}

__global__
__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel128_sparse_v2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev)
{
        int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+2))+lane;
        int offset2 = offset + MFACTOR;
        int offset3 = offset + MFACTOR*2;
        int offset4 = offset + MFACTOR*3;
        int i, j;


	//FTYPE r, r2;
	FTYPE o1 = vout[idx*sc+offset];
	FTYPE o2 = vout[idx*sc+offset2];
	FTYPE o3 = vout[idx*sc+offset3];
	FTYPE o4 = vout[idx*sc+offset4];
	int dummy = mcsr_cnt[idx/BH]*BH + ((idx&(BH-1))+1)*(mcsr_cnt[idx/BH+1] - mcsr_cnt[idx/BH]);
	int loc1 = mcsr_e[dummy-1], loc2 = mcsr_e[dummy];

        int buf; //FTYPE buf2;
        int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
        int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

	int jj=0, l;
	FTYPE r0=0.0f;

        buf = csr_e[loc1+lane];
	for(l=loc1; l<interm2; l+=4) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		int e2 = __shfl_sync(0xffffffff, buf, jj+1,MFACTOR)*sc;
		int e3 = __shfl_sync(0xffffffff, buf, jj+2,MFACTOR)*sc;
		int e4 = __shfl_sync(0xffffffff, buf, jj+3,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;
		FTYPE r2 = vin[e2+offset] * o1;
		r2 += vin[e2+offset2] * o2;
		r2 += vin[e2+offset3] * o3;
		r2 += vin[e2+offset4] * o4;
		FTYPE r3 = vin[e3+offset] * o1;
		r3 += vin[e3+offset2] * o2;
		r3 += vin[e3+offset3] * o3;
		r3 += vin[e3+offset4] * o4;
		FTYPE r4 = vin[e4+offset] * o1;
		r4 += vin[e4+offset2] * o2;
		r4 += vin[e4+offset3] * o3;
		r4 += vin[e4+offset4] * o4;

		int lv = (lane & 3);
		if(lv == 1) {
			FTYPE t = r2; r2 = r1; r1 = t;
			t = r4; r4 = r3; r3 = t;	
		} else if(lv == 2){
			FTYPE t = r3; r3 = r1; r1 = t;
			t = r4; r4 = r2; r2 = t;	
		} else if(lv == 3){
			FTYPE t = r4; r4 = r1; r1 = t;
			t = r3; r3 = r2; r2 = t;	
		} 

		r2 = __shfl_xor_sync(0xffffffff, r2, 1);
		r3 = __shfl_xor_sync(0xffffffff, r3, 2);
		r4 = __shfl_xor_sync(0xffffffff, r4, 3);
		r1 += r2+r3+r4;
		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);

		if(lane >= jj && lane < jj+4) {
			r0 = r1;
		}
                jj = ((jj+4)&(MFACTOR-1));
                if(jj == 0) {
		        buf = csr_e[l+4+lane];
			//atomicAdd(&ocsr_ev[l-28+lane], r0 * csr_ev[l-28+lane]);
			ocsr_ev[l-28+lane] = r0 * csr_ev[l-28+lane];
                }
	}
	if(interm2 < interm3) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		int e2 = __shfl_sync(0xffffffff, buf, jj+1,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;
		FTYPE r2 = vin[e2+offset] * o1;
		r2 += vin[e2+offset2] * o2;
		r2 += vin[e2+offset3] * o3;
		r2 += vin[e2+offset4] * o4;

		int lv = (lane & 1);
		if(lv == 1) {
			FTYPE t = r2; r2 = r1; r1 = t;
		}
 
		r2 = __shfl_xor_sync(0xffffffff, r2, 1);
		r1 += r2;
		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);
		r1 += __shfl_xor_sync(0xffffffff, r1, 2);

		if(lane >= jj && lane < jj+2) {
			r0 = r1;
		}

                jj = (jj+2);
	}
	if(interm3 < loc2) {
		int e1 = __shfl_sync(0xffffffff, buf, jj,MFACTOR)*sc;
		FTYPE r1 = vin[e1+offset] * o1;
		r1 += vin[e1+offset2] * o2;
		r1 += vin[e1+offset3] * o3;
		r1 += vin[e1+offset4] * o4;

		r1 += __shfl_xor_sync(0xffffffff, r1, 16);
		r1 += __shfl_xor_sync(0xffffffff, r1, 8);
		r1 += __shfl_xor_sync(0xffffffff, r1, 4);
		r1 += __shfl_xor_sync(0xffffffff, r1, 2);
		r1 += __shfl_xor_sync(0xffffffff, r1, 1);

		if(lane == jj) {
			r0 = r1;
		}

		jj = (jj+1);
	}
	l = loc1+(((l - loc1)>>5)<<5);
	if(lane < jj) {
		//atomicAdd(&ocsr_ev[l+lane], r0 * csr_ev[l+lane]);
		ocsr_ev[l+lane] = r0 * csr_ev[l+lane];
	}
}	




__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_ssparse(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev)
{
        int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
        int offset2 = offset + MFACTOR;
        int i, j;

        FTYPE r;//=0.0f;
        FTYPE r2;//=0.0f;
        int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

        int buf; FTYPE buf2;
        int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

        int jj=0, l;

        for(l=loc1; l<loc2; l++) {
                r = vin[csr_e[l]*sc+offset] * vout[idx*sc+offset];
                r2 = vin[csr_e[l]*sc+offset2] * vout[idx*sc+offset2];
                r += r2;
                r += __shfl_down_sync(0xffffffff, r, 16);
                r += __shfl_down_sync(0xffffffff, r, 8);
                r += __shfl_down_sync(0xffffffff, r, 4);
                r += __shfl_down_sync(0xffffffff, r, 2);
                r += __shfl_down_sync(0xffffffff, r, 1);
//printf("%f\n", r);
                if(lane == 0) {
                        //atomicAdd(&ocsr_ev[l], r*csr_ev[l]); // can be avoided
                        atomicAdd(&ocsr_ev[l], r*csr_ev[l]); // can be avoided
                }
        }
}

__global__
__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel128_dense_v2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, int *baddr, int *saddr, FTYPE *ocsr_ev)
{
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+2))+lane;
        int offset2 = offset + MFACTOR;
        int offset3 = offset + MFACTOR*2;
        int offset4 = offset + MFACTOR*3;
        int loop, i, j;

        __shared__ FTYPE sin[BW][MFACTOR];
        __shared__ FTYPE sin2[BW][MFACTOR];
        __shared__ FTYPE sin3[BW][MFACTOR];
        __shared__ FTYPE sin4[BW][MFACTOR];

        int warp_id = (threadIdx.x>>LOG_MFACTOR);

        int base_addr = baddr[blockIdx.x];
        int stride = saddr[blockIdx.x];

        for(i=warp_id;i<BW;i+=(DBSIZE>>LOG_MFACTOR)) {
                int hash = mcsr_list[blockIdx.x*BW + i];
                if(hash >= 0) {
                        sin[hash%BW][lane] = vin[hash*sc + offset];
                        sin2[hash%BW][lane] = vin[hash*sc + offset2];
                        sin3[hash%BW][lane] = vin[hash*sc + offset3];
                        sin4[hash%BW][lane] = vin[hash*sc + offset4];
                }
        }
        __syncthreads();

    for(i=warp_id;i<BH;i+=(DBSIZE>>LOG_MFACTOR)) {

        FTYPE o1 = vout[(base_addr*BH+i)*sc+offset];
        FTYPE o2 = vout[(base_addr*BH+i)*sc+offset2];
        FTYPE o3 = vout[(base_addr*BH+i)*sc+offset3];
        FTYPE o4 = vout[(base_addr*BH+i)*sc+offset4];


        int dummy = mcsr_cnt[base_addr]*BH + i*(mcsr_cnt[base_addr+1] - mcsr_cnt[base_addr]) + stride;
        int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

        int buf; //FTYPE buf2;
        //int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
        int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

        int jj=0, l;
        FTYPE r0=0.0f;

        buf = csr_e[loc1+lane];
        for(l=loc1; l<interm3; l+=2) {
                int e1 = (__shfl_sync(0xffffffff, buf, jj,MFACTOR)&(BW-1));
                int e2 = (__shfl_sync(0xffffffff, buf, jj+1,MFACTOR)&(BW-1));

                FTYPE r1 = sin[e1][lane] * o1;
                r1 += sin2[e1][lane] * o2;
                r1 += sin3[e1][lane] * o3;
                r1 += sin4[e1][lane] * o4;
                FTYPE r2 = sin[e2][lane] * o1;
                r2 += sin2[e2][lane] * o2;
                r2 += sin3[e2][lane] * o3;
                r2 += sin4[e2][lane] * o4;

                int lv = (lane & 1);
                if(lv == 1) {
                        FTYPE t = r2; r2 = r1; r1 = t;
                }

                r2 = __shfl_xor_sync(0xffffffff, r2, 1);
                r1 += r2;
//if(threadIdx.x == 0) printf("b: %f ", r1);
                r1 += __shfl_xor_sync(0xffffffff, r1, 16);
                r1 += __shfl_xor_sync(0xffffffff, r1, 8);
                r1 += __shfl_xor_sync(0xffffffff, r1, 4);
                r1 += __shfl_xor_sync(0xffffffff, r1, 2);
//if(threadIdx.x == 0) printf("a: %f ", r1);

                if(lane >= jj && lane < jj+2) {
                        r0 = r1;
                }

                jj = ((jj+2)&(MFACTOR-1));
                if(jj == 0) {
                        buf = csr_e[l+2+lane];
                        //atomicAdd(&ocsr_ev[l-30+lane], r0 * csr_ev[l-30+lane]);
                        ocsr_ev[l-30+lane] = r0 * csr_ev[l-30+lane];
//if(l-30+lane <32) printf("(%d: %f %f %f %f)", l-30+lane, ocsr_ev[l-30+lane], r0*csr_ev[l-30+lane], r0, csr_ev[l-30+lane]);
                }
        }


        if(interm3 < loc2) {
                int e1 = (__shfl_sync(0xffffffff, buf, jj,MFACTOR)&(BW-1));

                FTYPE r1 = sin[e1][lane] * o1;
                r1 += sin2[e1][lane] * o2;
                r1 += sin3[e1][lane] * o3;
                r1 += sin4[e1][lane] * o4;
                //r1 += vin[e1+offset3] * o3;
                //r1 += vin[e1+offset4] * o4;

                r1 += __shfl_xor_sync(0xffffffff, r1, 16);
                r1 += __shfl_xor_sync(0xffffffff, r1, 8);
                r1 += __shfl_xor_sync(0xffffffff, r1, 4);
                r1 += __shfl_xor_sync(0xffffffff, r1, 2);
                r1 += __shfl_xor_sync(0xffffffff, r1, 1);

                if(lane == jj) {
                        r0 = r1;
                }

                jj = (jj+1);
        }
        l = loc1+(((l - loc1)>>5)<<5);
        if(lane < jj) {
                //atomicAdd(&ocsr_ev[l+lane], r0 * csr_ev[l+lane]);
                ocsr_ev[l+lane] = r0 * csr_ev[l+lane];
        }



    }
}




#define ITER (128/128) * 10

__global__ void dense_block_detect(int *csr_v, int *mcsr_chk, int *csr_e0, int *flag)
{
	int i;
	int lb = csr_v[blockIdx.x*BH];
	int ub = csr_v[(blockIdx.x+1)*BH];
	//__shared__ short scr_pad[SC_SIZE];
	__shared__ int scr_pad[SC_SIZE];

	for(i=threadIdx.x; i<SC_SIZE; i+=blockDim.x) {
		scr_pad[i] = 0;
	}
	__syncthreads();
	for(i=lb+threadIdx.x; i<ub; i+=blockDim.x) {
		int key = (csr_e0[i]&(SC_SIZE-1));
		if(scr_pad[key] < THRESHOLD) atomicAdd(&scr_pad[key], 1);
	}
	__syncthreads();
	int r=0;
	for(i=threadIdx.x; i<SC_SIZE; i+=blockDim.x) {
		if(scr_pad[i] >= THRESHOLD) r++;
	}
	__syncthreads();
	r += __shfl_down_sync(0xffffffff, r, 16);
	r += __shfl_down_sync(0xffffffff, r, 8);
	r += __shfl_down_sync(0xffffffff, r, 4);
	r += __shfl_down_sync(0xffffffff, r, 2);
	r += __shfl_down_sync(0xffffffff, r, 1);
	if((threadIdx.x & 31) == 0) scr_pad[threadIdx.x>>5] = r;
	__syncthreads();
	if(threadIdx.x == 0) {
		for(i=1; i<BH/32; i++)
			r += scr_pad[i];
		if(r >= MIN_OCC) {
			mcsr_chk[blockIdx.x] = 1;
			if(flag[blockIdx.x&127] == 0) flag[blockIdx.x&127] = 1;
		}
	}
}


__global__ void simple_mcsr_cnt(int npanel, int *mcsr_cnt)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < npanel) mcsr_cnt[idx] = idx;
}

__global__ void csr_pivot_gen(int npanel, int *csr_v, int *csr_pivot)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < npanel) {
		csr_pivot[idx] = csr_v[(idx)*BH];
	}
}

__global__ void csr_pnt_gen(int ne, int *csr_e0, int *key, STYPE *key2, int *val)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < ne) {
		key[idx] = csr_e0[idx];
		key2[idx] = 30000; 
		val[idx] = idx;
	}
}

#define MCSR_CNT_SIZE (1024)
__global__ void mcsr_cnt_calc(int *csr_pivot, int *key, int *mcsr_cnt, int *mcsr_chk)
{
	if(mcsr_chk[blockIdx.x] == 0) return;
	int lb = csr_pivot[blockIdx.x]+THRESHOLD-1;
	int ub = csr_pivot[blockIdx.x+1];
	__shared__ int age[BW];
	__shared__ int occ[MCSR_CNT_SIZE];
	for(int i=threadIdx.x; i<BW; i+=blockDim.x) {
		age[i] = 0;
	}
	for(int i=threadIdx.x; i<MCSR_CNT_SIZE; i+=blockDim.x) {
		if(i > 0) occ[i] = 0;
		else occ[i] = BW;
	}
	__syncthreads();

	for(int i=lb+threadIdx.x; i<ub; i+=blockDim.x) {
		if(i == ub - 1 || key[i] != key[i+1]) {
			if(key[i] == key[i-(THRESHOLD-1)]) {
				int hash = atomicAdd(&age[key[i]&(BW-1)], 1);
				atomicAdd(&occ[hash+1], 1);
			}
		}
	}
	__syncthreads();
	if(threadIdx.x < MCSR_CNT_SIZE-1 && occ[threadIdx.x] >= MIN_OCC && occ[threadIdx.x+1] < MIN_OCC) {
		mcsr_cnt[blockIdx.x+1] = threadIdx.x;
	} 

}

#define LIST_CANDI (1024*4)

__global__ void key2_marking(int *csr_pivot, int *key, STYPE *key2, int *val, int *mcsr_cnt, int *mcsr_list, int *baddr, int *saddr, int *mcsr_chk)
{
	if(mcsr_chk[blockIdx.x] == 0) return;
	int lb = csr_pivot[blockIdx.x]+THRESHOLD-1;
	int ub = csr_pivot[blockIdx.x+1];
	int uub = lb+CEIL(ub-lb,1024)*1024;
	int bloc = (mcsr_cnt[blockIdx.x] - blockIdx.x)*BW;
	int limit = mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x] - 1;

	__shared__ int age[BW];
	__shared__ int list[LIST_CANDI];  
	__shared__ short list2[LIST_CANDI];
	__shared__ int listp;
	for(int i=threadIdx.x; i<BW; i+=blockDim.x) {
		age[i] = 0;
	}
	__syncthreads();
	for(int i=threadIdx.x; i<limit; i+=blockDim.x) {
		baddr[mcsr_cnt[blockIdx.x]-blockIdx.x+i] = blockIdx.x;
		saddr[mcsr_cnt[blockIdx.x]-blockIdx.x+i] = threadIdx.x;
	}
    for(int i0=lb+threadIdx.x; i0<uub; i0+=LIST_CANDI*THRESHOLD) {
	if(threadIdx.x == 0) listp=0;
	__syncthreads();
	for(int i=i0; i<MIN(i0+LIST_CANDI*THRESHOLD,ub); i+=blockDim.x) {
		if(i == ub - 1 || key[i] != key[i+1]) {
			if(key[i] == key[i-(THRESHOLD-1)]) {
				int width = (key[i]&(BW-1));
				int depth = atomicAdd(&age[width], 1);
				if(depth < limit) {
					mcsr_list[bloc + depth*BW + width] = key[i];
					int p = atomicAdd(&listp, 1);
					list[p] = i;
					list2[p] = depth;		
				} 			

			}
		}
	}
	__syncthreads();
#define LLF (8)
#define LOG_LLF (3)
	for(int i=(threadIdx.x>>LOG_LLF); i<listp; i+=(blockDim.x>>LOG_LLF)) {
		int p = list[i];
		int depth = list2[i];
		int width = (key[p]&(BW-1));
		for(int k=p-(threadIdx.x&(LLF-1)); k >= csr_pivot[blockIdx.x] && key[k] == key[p]; k-=LLF) {
			key2[val[k]] = depth;
		} 	
	}
	__syncthreads();
    }
}

__global__ void fill_val(int ne, int *val)
{
	int idx = blockIdx.x*blockDim.x*4 + threadIdx.x;
	int idx2 = idx + blockDim.x;
	int idx3 = idx + blockDim.x*2;
	int idx4 = idx + blockDim.x*3;
	if(idx4 < ne) {
		val[idx] = idx;
		val[idx2] = idx2;
		val[idx3] = idx3;
		val[idx4] = idx4;
	} else if(idx3 < ne) {
		val[idx] = idx;
		val[idx2] = idx2;
		val[idx3] = idx3;
	} else if(idx2 < ne) {
		val[idx] = idx;
		val[idx2] = idx2;
	} else if(idx < ne) {
		val[idx] = idx;
	}
}

__global__ void fill_mcsre(int *csr_v, int *mcsr_cnt, STYPE *key2, int *mcsr_e, int *rmv)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int delta = mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x];
	int bidx = mcsr_cnt[blockIdx.x]*BH + delta*threadIdx.x;
	int i = csr_v[idx];
	//int lb, ub=key2[i];
        int kk = MIN(key2[i], delta-1);
        if(i == csr_v[idx+1]) kk = delta-1;
        for(int j = 0; j<=kk; j++)
                mcsr_e[bidx+j] = csr_v[idx];

	for(; i<csr_v[idx+1]; i++) {
		int lb = key2[i], ub = key2[i+1];
		//lb = ub; ub = key2[i+1];
		if(lb == 30000) break;
		if(i == csr_v[idx+1]-1 || ub >= delta) ub = delta-1;
		for(int j = lb+1; j<=ub; j++) {
			mcsr_e[bidx+j] = i+1;
		}
		
	}
	int r = (csr_v[idx+1] - mcsr_e[bidx+delta-1]);
	r += __shfl_down_sync(0xffffffff, r, 16);
	r += __shfl_down_sync(0xffffffff, r, 8);
	r += __shfl_down_sync(0xffffffff, r, 4);
	r += __shfl_down_sync(0xffffffff, r, 2);
	r += __shfl_down_sync(0xffffffff, r, 1);
	if((threadIdx.x&31) == 0) atomicAdd(&rmv[(idx>>5)&127], r);
}

__global__ void porting(int ne, int *val, int *csr_e0, FTYPE *csr_ev0, int *csr_e, FTYPE *csr_ev)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < ne) {
		int k = val[idx];
		csr_e[idx] = csr_e0[k];
		csr_ev[idx] = csr_ev0[k];	
	}
}

__global__ void cal_vari(int nr, double avg, int *mcsr_cnt, int *mcsr_e, double *vari, int *special_bb)
{
	int idx = (mcsr_cnt[blockIdx.x]*BH) + (mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x])*(threadIdx.x+1);
	int i2 = mcsr_e[idx] - mcsr_e[idx-1];
	double r = ((double)i2 - avg);
	double r2 = r*r;

	r2 += __shfl_down_sync(0xffffffff, r2, 16);
	r2 += __shfl_down_sync(0xffffffff, r2, 8);
	r2 += __shfl_down_sync(0xffffffff, r2, 4);
	r2 += __shfl_down_sync(0xffffffff, r2, 2);
	r2 += __shfl_down_sync(0xffffffff, r2, 1);
	i2 /= STHRESHOLD;
	i2 += __shfl_down_sync(0xffffffff, i2, 16);	
	i2 += __shfl_down_sync(0xffffffff, i2, 8);	
	i2 += __shfl_down_sync(0xffffffff, i2, 4);	
	i2 += __shfl_down_sync(0xffffffff, i2, 2);	
	i2 += __shfl_down_sync(0xffffffff, i2, 1);	
	if((threadIdx.x&31) == 0) {
		atomicAdd(&vari[((blockIdx.x*blockDim.x)>>5)&127], r2);
		if(i2 > 0) atomicAdd(&special_bb[((blockIdx.x*blockDim.x)>>5)&127], i2);
	}
}


__global__ void make_special(int *mcsr_cnt, int *mcsr_e, int *special, int *special2, int *scnt)
{
	int idx = (mcsr_cnt[blockIdx.x]*BH) + (mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x])*(threadIdx.x+1);
	int i2 = (mcsr_e[idx] - mcsr_e[idx-1])/STHRESHOLD;
	if(i2 > 0) {
		int k = atomicAdd(&scnt[0], i2);
		for(int i=k;i<k+i2;i++) {
			special[i] = blockIdx.x*blockDim.x+threadIdx.x;
			special2[i] = STHRESHOLD*(i-k);
		}
	}
}

void process()
{

	int i, j;

	int *_csr_v; int *_csr_e0; FTYPE *_csr_ev0;
	int *_csr_e; FTYPE *_csr_ev;
	FTYPE *_ocsr_ev;

        cudaMalloc((void **) &_csr_v, sizeof(int)*(nr+1));
        cudaMalloc((void **) &_csr_e0, sizeof(int)*ne+256);
        cudaMalloc((void **) &_csr_ev0, sizeof(FTYPE)*ne+256);
        cudaMalloc((void **) &_ocsr_ev, sizeof(FTYPE)*ne+256);

        cudaMemset(_csr_v, 0, sizeof(int)*(nr+1));
        cudaMemset(_csr_e0, 0, sizeof(int)*ne+256);
        cudaMemset(_csr_ev0, 0, sizeof(FTYPE)*ne+256);
        cudaMemset(_ocsr_ev, 0, sizeof(FTYPE)*ne+256);

        cudaMemcpy(_csr_v, csr_v, sizeof(int)*(nr+1), cudaMemcpyHostToDevice);
        cudaMemcpy(_csr_e0, csr_e0, sizeof(int)*(ne+1), cudaMemcpyHostToDevice);
        cudaMemcpy(_csr_ev0, csr_ev0, sizeof(FTYPE)*ne, cudaMemcpyHostToDevice);



	int *_mcsr_cnt;
	int *_mcsr_chk;
	int *_mcsr_e;
	int *_mcsr_list;

        cudaMalloc((void **) &_mcsr_cnt, sizeof(int)*(npanel+1));
        cudaMalloc((void **) &_mcsr_chk, sizeof(int)*(npanel+1));

	//TODO opt-space
	cudaMalloc((void **) &_mcsr_e, sizeof(int)*ne+256);
	cudaMalloc((void **) &_mcsr_list, sizeof(int)*ne+256);

	cudaMemset(_mcsr_cnt, 0, sizeof(int)*(npanel+1));
	cudaMemset(_mcsr_chk, 0, sizeof(int)*(npanel+1));
	cudaMemset(_mcsr_e, 0, sizeof(int)*ne+256);
	cudaMemset(_mcsr_list, -1, sizeof(int)*ne+256);

	int *_baddr, *_saddr;
	int *_special, *_special2;

	/////
        FTYPE *vin, *_vin, *vout, *_vout;
        FTYPE *vout_gold;
        vin = (FTYPE *)malloc(sizeof(FTYPE)*nc*sc);
        vout = (FTYPE *)malloc(sizeof(FTYPE)*nr*sc);
        vout_gold = (FTYPE *)malloc(sizeof(FTYPE)*ne+256);

        cudaError_t err = cudaSuccess;

        err = cudaMalloc((void **) &_vin, sizeof(FTYPE)*nc*sc);
        if(err != 0) exit(0);
        err = cudaMalloc((void **) &_vout, sizeof(FTYPE)*nr*sc);
        if(err != 0) exit(0);

        for(i=0;i<nc*sc;i++) {
                vin[i] = (FTYPE)(rand()%1048576)/1048576;
#ifdef SIM_VALUE
		vin[i] = 1;
#endif
        }
        for(i=0;i<nr*sc;i++) {
                vout[i] = (FTYPE)(rand()%1048576)/1048576;
#ifdef SIM_VALUE
                vout[i] = 1;
#endif
        }
        cudaMemcpy(_vin, vin, sizeof(FTYPE)*nc*sc, cudaMemcpyHostToDevice);
        cudaMemcpy(_vout, vout, sizeof(FTYPE)*nr*sc, cudaMemcpyHostToDevice);


	int *_rmv;
	cudaMalloc((void **) &_rmv, sizeof(int)*128);
	cudaMemset(_rmv, 0, sizeof(int)*128);
	int *rmv = (int *)malloc(sizeof(int)*128);

	double *_vari;
	cudaMalloc((void **) &_vari, sizeof(double)*128);
	cudaMemset(_vari, 0, sizeof(double)*128);
	double *vari0 = (double *)malloc(sizeof(double)*128);

	int *_special_bb;
	cudaMalloc((void **) &_special_bb, sizeof(int)*128);
	cudaMemset(_special_bb, 0, sizeof(int)*128);
	int *special_bb = (int *)malloc(sizeof(int)*128);

	int *_scnt;
	cudaMalloc((void **) &_scnt, sizeof(int));
	cudaMemset(_scnt, 0, sizeof(int));

	int detect_nb = CEIL(nr, BH);
	int d_flag[128];
	int *_d_flag;
	cudaMalloc((void **) &_d_flag, sizeof(int)*128);
	cudaMemset(_d_flag, 0, sizeof(int)*128);

	int pivot_gen_nb = CEIL(npanel+1, 128);
	int *_csr_pivot;
        cudaMalloc((void **) &_csr_pivot, sizeof(int)*(npanel+1));
	cudaMemset(_csr_pivot, 0, sizeof(int)*(npanel+1));

	int pnt_gen_nb = CEIL(ne+1, 128);
	int *_key; STYPE *_key2;
	int *_val;
        cudaMalloc((void **) &_key, sizeof(int)*ne+256);
        cudaMalloc((void **) &_key2, sizeof(STYPE)*ne+256);
        cudaMalloc((void **) &_val, sizeof(int)*ne+256);

	int fill_nb = CEIL(ne, 128*4);
	int mcsr_nb = CEIL(nr, BH);
	int port_nb = CEIL(ne, 128);

#define MCSR_CNT_TBSIZE (1024)
	int mcsr_cnt_nb = CEIL(npanel, MCSR_CNT_TBSIZE);

	int s_npanel = CEIL(npanel, 8);

        cudaStream_t stream1, stream2, stream3;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);

	// pre-process
	float ptot_ms;
	cudaEvent_t pevent1, pevent2;
	cudaEventCreate(&pevent1);
	cudaEventCreate(&pevent2);
      	cudaDeviceSynchronize();
	cudaEventRecord(pevent1,0);

	dense_block_detect<<<detect_nb, BH>>>(_csr_v, _mcsr_chk, _csr_e0, _d_flag);
	
	cudaMemcpy(d_flag, _d_flag, sizeof(int)*128, cudaMemcpyDeviceToHost); 

	for(int i=1;i<128;i++) { 
		d_flag[0] += d_flag[i];
	}


  if(d_flag[0] == 0) {
	num_dense = 0;
	avg = (double)ne / nr;
	_mcsr_e = _csr_v;
	_csr_e = _csr_e0;
	_csr_ev = _csr_ev0;
	simple_mcsr_cnt<<<pivot_gen_nb, 128>>>(npanel+1, _mcsr_cnt);
  } else {

	csr_pivot_gen<<<pivot_gen_nb, 128, 0, stream1>>>(npanel+1, _csr_v, _csr_pivot);
	csr_pnt_gen<<<pnt_gen_nb, 128, 0, stream2>>>(ne+1, _csr_e0, _key, _key2, _val);

#ifdef PP_TIME
float ptot_ms0;
cudaEvent_t pevent10, pevent20;
cudaEventCreate(&pevent10);
cudaEventCreate(&pevent20);
cudaDeviceSynchronize();
cudaEventRecord(pevent10,0);
#endif
	bb_segsort(_key, _val, ne, _csr_pivot, npanel);
#ifdef PP_TIME
cudaEventRecord(pevent20,0);
cudaEventSynchronize(pevent10);
cudaEventSynchronize(pevent20);
cudaEventElapsedTime(&ptot_ms0, pevent10, pevent20);
cudaDeviceSynchronize();
printf("it1 : %f\n", ptot_ms0);
#endif
	mcsr_cnt_calc<<<npanel, MCSR_CNT_TBSIZE>>>(_csr_pivot, _key, _mcsr_cnt, _mcsr_chk);

	int *ttt=(int *)malloc(sizeof(int)*(npanel+1)); 	
	cudaMemcpy(ttt, _mcsr_cnt, sizeof(int)*(npanel+1), cudaMemcpyDeviceToHost);
	for(int i=1;i<=npanel;i++) {
		ttt[i] += ttt[i-1]+1;
	}
	num_dense = ttt[npanel] - npanel;
	cudaMemcpy(_mcsr_cnt, ttt, sizeof(int)*(npanel+1), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &_baddr, sizeof(int)*ttt[npanel]);
	cudaMalloc((void **) &_saddr, sizeof(int)*ttt[npanel]);

	key2_marking<<<npanel, MCSR_CNT_TBSIZE>>>(_csr_pivot, _key, _key2, _val, _mcsr_cnt, _mcsr_list, _baddr, _saddr, _mcsr_chk);	

	fill_val<<<fill_nb, 128>>>(ne, _val);
#ifdef PP_TIME
float ptot_ms1;
cudaEvent_t pevent11, pevent21;
cudaEventCreate(&pevent11);
cudaEventCreate(&pevent21);
cudaDeviceSynchronize();
cudaEventRecord(pevent11,0);
#endif
	bb_segsort(_key2, _val, ne, _csr_v, nr);
#ifdef PP_TIME
cudaEventRecord(pevent21,0);
cudaEventSynchronize(pevent11);
cudaEventSynchronize(pevent21);
cudaEventElapsedTime(&ptot_ms1, pevent11, pevent21);
cudaDeviceSynchronize();
printf("it2 : %f\n", ptot_ms1);
#endif
	fill_mcsre<<<mcsr_nb, BH>>>(_csr_v, _mcsr_cnt, _key2, _mcsr_e, _rmv);
	
	cudaMemcpy(rmv, _rmv, sizeof(int)*128, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_mcsr_e[BH*(num_dense+npanel)], &ne, sizeof(int), cudaMemcpyHostToDevice);
	for(int i=1;i<128;i++) 
		rmv[0] += rmv[i];
	avg = (double)rmv[0] / nr;

	cudaFree(_key);
	cudaFree(_key2);

        cudaMalloc((void **) &_csr_e, sizeof(int)*ne+256);
        cudaMalloc((void **) &_csr_ev, sizeof(FTYPE)*ne+256);

	porting<<<port_nb, 128>>>(ne, _val, _csr_e0, _csr_ev0, _csr_e, _csr_ev);
  }
	cal_vari<<<npanel, BH>>>(nr, avg, _mcsr_cnt, _mcsr_e, _vari, _special_bb);
	cudaMemcpy(vari0, _vari, sizeof(double)*128, cudaMemcpyDeviceToHost);
	for(int i=1;i<128;i++) { 
		vari0[0] += vari0[i];
	}
	vari = (double)vari0[0] / nr;

//fprintf(stderr, "avg : %f, vari : %f, num_dense : %d\n", avg, vari, num_dense);

	if(vari >= 200) {
		cudaMemcpy(special_bb, _special_bb, sizeof(int)*128, cudaMemcpyDeviceToHost);
		for(int i=0;i<128;i++) { 
			special_p += special_bb[i];
		}
		cudaMalloc((void **) &_special, sizeof(int)*special_p);
        	cudaMalloc((void **) &_special2, sizeof(int)*special_p);
		make_special<<<npanel, BH>>>(_mcsr_cnt, _mcsr_e, _special, _special2, _scnt);
	}

	cudaEventRecord(pevent2,0);
	cudaEventSynchronize(pevent1);
	cudaEventSynchronize(pevent2);
	cudaEventElapsedTime(&ptot_ms, pevent1, pevent2);
	cudaDeviceSynchronize();
	
//printf("time : %f\n", ptot_ms);
// printf("%f,", ptot_ms);

	// process
	dim3 s_gridsize(nr/SBF, 1, CEIL(sc, MFACTOR*4));
	dim3 s_blocksize(SBSIZE, 1, 1);
	dim3 ss_gridsize(nr/SBF, 1, CEIL(sc, MFACTOR*2));
	dim3 ss_blocksize(SBSIZE, 1, 1);
	dim3 d_gridsize(num_dense, 1, CEIL(sc, MFACTOR*4));
	dim3 d_blocksize(DBSIZE, 1, 1);
	dim3 s_gridsizeh(special_p, 1, CEIL(sc, MFACTOR*4));
	dim3 s_blocksizeh(SPBSIZE, 1, 1);


	float tot_ms;
	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

if (ne/nc < 6 && vari < 40) { 
      	cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
for(int ik=0;ik<ITER;ik++) {
	// kernel fun
        spmv_kernel64_ssparse<<<ss_gridsize, ss_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e, _csr_ev, _mcsr_cnt, _mcsr_e, _mcsr_list, _vin, _vout, _ocsr_ev);
}
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);
	cudaDeviceSynchronize();

} else if (vari < 200) {
       	cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
for(int ik=0;ik<ITER;ik++) {
	// kernel fun
        spmv_kernel128_sparse_v2<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e, _csr_ev, _mcsr_cnt, _mcsr_e, _mcsr_list, _vin, _vout, _ocsr_ev);
        spmv_kernel128_dense_v2<<<d_gridsize, d_blocksize, 0, stream2>>>(sc, _csr_v, _csr_e, _csr_ev, _mcsr_cnt, _mcsr_e, _mcsr_list, _vin, _vout, _baddr, _saddr, _ocsr_ev);
}	
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);
	cudaDeviceSynchronize();
} else {
	cudaDeviceSynchronize();
	cudaEventRecord(event1,0);

for(int ik=0;ik<ITER;ik++) {
	// kernel fun
        spmv_kernel128_sparse_v2l<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e, _csr_ev, _mcsr_cnt, _mcsr_e, _mcsr_list, _vin, _vout, _ocsr_ev);
	spmv_kernel128_sparse_v2h<<<s_gridsizeh, s_blocksizeh, 0, stream3>>>(sc, _csr_v, _csr_e, _csr_ev, _mcsr_cnt, _mcsr_e, _mcsr_list, _vin, _vout, _ocsr_ev, _special, _special2);
        spmv_kernel128_dense_v2<<<d_gridsize, d_blocksize, 0, stream2>>>(sc, _csr_v, _csr_e, _csr_ev, _mcsr_cnt, _mcsr_e, _mcsr_list, _vin, _vout, _baddr, _saddr, _ocsr_ev);
}	
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);
	cudaDeviceSynchronize();

}
//fprintf(stdout, "%d,%d,%d,%d,", nr, nc, ne, mne);
//        fprintf(stdout, "%f,%f,", tot_ms,(double)ne*2*sc/tot_ms/1000000);
        // fprintf(stdout, "%f,%f,", (double)ITER*(double)ne*2*sc/tot_ms/1000000,(double)ptot_ms/tot_ms);
	// FILE *fpo = fopen("SDDMM_GPU_SP.out", "a");
        // fprintf(fpo, "%f,", (double)ITER*(double)ne*2*sc/tot_ms/1000000);

		printf(",%f,%f\n",tot_ms,(double)ITER*(double)ne*2*sc/tot_ms/1000000);
	

        cudaMemcpy(ocsr_ev, _ocsr_ev, sizeof(FTYPE)*ne, cudaMemcpyDeviceToHost);

// #define VALIDATE
#if defined VALIDATE
	cudaMemcpy(csr_e0, _csr_e, sizeof(int)*ne, cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_ev0, _csr_ev, sizeof(FTYPE)*ne, cudaMemcpyDeviceToHost);

        //validate
        for(i=0;i<ne;i++) {
                vout_gold[i] = 0.0f;
        }
        for(i=0;i<nr;i++) {
                for(j=csr_v[i]; j<csr_v[i+1]; j++) {
                        int k;
                        for(k=0; k<sc; k++) {
                                vout_gold[j] += vin[sc*csr_e0[j] + k] * vout[sc*i + k];
                        }
                        vout_gold[j] *= csr_ev0[j];
                }
        }
        int num_diff=0;
        for(i=0;i<ne;i++) {
                FTYPE p1 = vout_gold[i]; FTYPE p2 = ocsr_ev[i];
		//if (ne/nc >= 6 || vari >= 40) p1 *= ITER;
                if(p1 < 0) p1 *= -1;
                if(p2 < 0) p2 *= -1;
                FTYPE diff;
                diff = p1 - p2;
                if(diff < 0) diff *= -1;
                if(diff / MAX(p1,p2) > 0.01) {
                        if(num_diff < 20*1*1) fprintf(stdout, "\n%d %f %f", i, vout[i], vout_gold[i]);
			//if(vout[i] < vout_gold[i]) fprintf(stdout, "%d %f %f\n", i, vout[i], vout_gold[i]);

                        num_diff++;
                }
        }
//      fprintf(stdout, "num_diff : %d\n", num_diff);
        //fprintf(stdout, "%f,", (double)num_diff/ne*100);
        // fprintf(fpo, "%f,", (double)num_diff/ne*100);

//fprintf(stdout, "X(%f %f)\n", x1, x2);
//      fprintf(stdout, "ne : %d\n", gold_ne);
#endif
//	fprintf(stdout, "\n");
        free(vin); free(vout); cudaFree(_vin); cudaFree(_vout);
        free(vout_gold);
	// fclose(fpo);
	// printf("\n");
//printf("st ; %d\n", SSTRIDE);
}

int main(int argc, char **argv)
{
	// cudaSetDevice(1);
	ready(argc, argv);
	//gen_structure();
	process();
}

