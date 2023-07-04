#include <iostream>
#include <fstream>
#include <sys/io.h>

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "matrix_utils.h"

using namespace std;

int main() {
    string data_dir = "/data/pm/sparse_matrix/data";

	ofstream fout("mat_info1.txt",ios::out);
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat(data_dir.c_str(), &s );
	if(!S_ISDIR( s.st_mode ) ) 
    {
		cout<<"dir_name is not a valid directory !"<<endl;
		return;
	}
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir( data_dir.c_str() );
	if( NULL == dir ) 
    {
		cout<<"Can not open dir "<<data_dir<<endl;
		return;
	}

    double gflops = 0.0f;

	int pdf[5000 + 10];

	for(int i = 0; i < 5010; ++i) {
		pdf[i] = 0;
	}
	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL )
	{
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
		// cout<<filename ->d_name <<endl;

        string file_path = data_dir +"/"+ filename->d_name + "/" + filename->d_name + ".mtx";
        // file_path = "/data/pm/sparse_matrix/data/cage15/cage15.mtx";

        // cout<<file_path<<endl;

        // break;
        fout<<filename->d_name<<" ";
        cout<<filename->d_name<<endl;

        SPC::SparseMatrix sm1(file_path,SPC::SORTED,1);
        

		int nrows = sm1.Rows();

		double avg = (double) sm1.Nonzeros() / (double) nrows;
		double var = 0;
		int maxr = 0;
		int minr = 0;


        int * row_offset_h = sm1.RowOffsets();
        int * row_indices_h = sm1.RowIndices();

        // for(int i=0; i < 10; ++i)
        //     cout<<"  row["<<row_indices_h[i]<<"] : "<<row_offset_h[row_indices_h[i]+1] - row_offset_h[row_indices_h[i]]<<endl;


		for(int i = 0; i < nrows; ++i) {
			int rnnz = row_offset_h[i+1] - row_offset_h[i];

			var += ( (double) rnnz - avg ) * ( (double) rnnz - avg )  /(double) nrows;
			if(rnnz < 32 * 5001) {
				int idx = rnnz / 32;
 				pdf[idx] ++;
			} else {
				pdf[5001] ++; 
			}

			if(maxr < rnnz) 
				maxr = rnnz;
			if(minr > rnnz)
				minr = rnnz;
			// cout<<rnnz<<endl;
		}

		int polar = maxr - minr;

		fout<< sm1.Rows() <<" " << sm1.Columns()<< " " << sm1.Nonzeros()<<" "<<avg<<" "<<var<<" "<<polar<<endl;

        cout<< sm1.Rows() <<" " << sm1.Columns()<< " " << sm1.Nonzeros()<<" "<<avg<<" "<<var<<" "<<polar<<endl;
	}

	for(int i = 0; i < 5002; ++i) {
		fout<<pdf[i]<<" ";
		cout<<pdf[i]<<" ";
	}
	fout<<endl;
	cout<<endl;
	fout.close();
    return 0;
}