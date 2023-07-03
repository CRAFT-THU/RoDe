mkdir data2
python download.py
cd data2
# wget https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian14.tar.gz

find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz
cp ../conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=`echo ${i} | sed -e "s/\///g"`
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

rm conv
rm conv.c
cd ..