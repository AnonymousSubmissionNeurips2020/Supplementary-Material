echo $PASSWORD | sudo -S apt-get install unzip
echo $PASSWORD | sudo -S apt-get install bzip2

wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00220/Relation%20Network%20(Directed).data" 
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00424/2014%20and%202015%20CSM%20dataset.xlsx" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv" 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip" 
unzip Facebook_metrics.zip 
rm Facebook_metrics.zip 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00335/online_video_dataset.zip" 
unzip online_video_dataset.zip 
rm online_video_dataset.zip 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip" 
unzip OnlineNewsPopularity.zip 
rm OnlineNewsPopularity.zip 
wget "http://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip" 
unzip slice_localization_data.zip 
rm slice_localization_data.zip 
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip" 
unzip UJIndoorLoc.zip 
rm UJIndoorLoc.zip 

wget "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2"
wget "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2"
bzip2 -dk news20.scale.bz2
bzip2 -dk news20.t.scale.bz2