# rsync -avzh -e 'ssh -p 22112' --progress Documents/graduation_project/ guochx@10.15.49.6:~/graduation_project/ 
rsync -avzh -e 'ssh -p 22112' --progress "Documents/graduation_project/*.py" guochx@10.15.49.6:/hpc/data/home/bme/guochx/graduation_project/ 
