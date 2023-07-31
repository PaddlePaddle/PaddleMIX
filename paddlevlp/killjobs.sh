ps -ef|grep 'from multiprocessing'|awk {'print $2'}|xargs kill -9
ps -ef|grep 'run_pretrain_dist.py'|awk {'print $2'}|xargs kill -9
ps -ef|grep './dist_train.sh'|awk {'print $2'}|xargs kill -9
# ps -ef|grep 'tensorboard'|awk {'print $2'}|xargs kill -9
ps -ef|grep 'visualdl'|awk {'print $2'}|xargs kill -9
