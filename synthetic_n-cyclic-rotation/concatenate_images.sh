	


filename='n-element-GAN.png'


# sudo apt install imagemagick-6.q16 

montage \
./result/fake_process_n,2,P,0.00.png ./result/fake_process_n,2,P,0.17.png ./result/fake_process_n,2,P,0.33.png ./result/fake_process_n,2,P,0.50.png ./result/fake_process_n,2,P,1.00.png \
./result/fake_process_n,3,P,0.00.png ./result/fake_process_n,3,P,0.22.png ./result/fake_process_n,3,P,0.44.png ./result/fake_process_n,3,P,0.67.png ./result/fake_process_n,3,P,1.00.png \
./result/fake_process_n,4,P,0.00.png ./result/fake_process_n,4,P,0.25.png ./result/fake_process_n,4,P,0.50.png ./result/fake_process_n,4,P,0.75.png ./result/fake_process_n,4,P,1.00.png \
./result/fake_process_n,5,P,0.00.png ./result/fake_process_n,5,P,0.27.png ./result/fake_process_n,5,P,0.53.png ./result/fake_process_n,5,P,0.80.png ./result/fake_process_n,5,P,1.00.png \
-geometry +5+2 ./$filename



echo $filename


