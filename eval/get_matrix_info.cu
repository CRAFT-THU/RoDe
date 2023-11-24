#include "cuda_runtime.h"
#include "matrix_utils.h"

#include <sys/io.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

using namespace std;
using namespace SPC;


int main(int argc,char **argv) {

    // cudaSetDevice(1);

    string file_path;
    if(argc < 2) {
        cout<<"No file path"<<endl;
        return 0;
    }
    else {
        file_path = argv[1];
    }

    double gflops = 0.0f;

    // cout<<file_path<<endl;

    SPC::SparseMatrix sm1(file_path,SPC::SORTED,1);

    // nr, nc, nnz

    printf(", %d, %d, %d\n",sm1.Rows(),sm1.Columns(),sm1.Nonzeros());
	
    return 0;
}
// 
