# K-means-cluster
Finland Geographical Data

Only requirements for the code to run are the two geography data files, `FinlandWhole.txt` and `JoensuuRegion.txt`.

The code initializes centroids smartly by choosing centroids far apart and spread out across the data. The method is to randomly choose the first centroid, and choose the next centroid to be the data point with the largest minimum distance to previous centroids. 

The number of `ITERATIONS` is set to 10 since that was enough for convergence. For each dataset, "Finland" and "Joensuu Region", the code plots the cluster results (`_cluster.png`), mean cluster distance (MCD) for k=4 over iterations (`_mcd.png`), and MCD for various k’s (`_variousk.png`). I chose to test for k values [4, 8, 12, 20]. 
