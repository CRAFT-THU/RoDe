#!/bin/bash
cd ..

mkdir data
cd data
wget https://www.cise.ufl.edu/research/sparse/MM/Priebel/192bit.tar.gz
wget https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/wv2010.tar.gz
wget https://www.cise.ufl.edu/research/sparse/MM/Ronis/xenon2.tar.gz
wget https://www.cise.ufl.edu/research/sparse/MM/VanVelzen/Zd_Jac2_db.tar.gz
wget https://www.cise.ufl.edu/research/sparse/MM/Zhao/Zhao2.tar.gz
wget https://www.cise.ufl.edu/research/sparse/MM/Andrianov/mip1.tar.gz
find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz
cp ../script/conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

rm conv
rm conv.c
cd ..

