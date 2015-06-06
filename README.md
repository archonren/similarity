# similarity
find the most similar user
## useage
run in command line:
python main.py userkey1 uesrkey2 userkey3 (optional: --k=somenumber --model_path=path to model file --user_data_path = path to user data --verbosity = True or False, show progress)
example:
python main.py 23456 56447 --k=2000

and it will give a list of top N (default 3) users that are most similar to the user
##warning
running this code first time is very very slow because it needs to create all the required model files, once those files are created, the code will be much faster. 
## todo
1.N, model path and user_data should be able to input via command line （done）

2.all data are pickel file, need to find a way to convert into json (json cannot take numpy array as value)

3.some tags cannot be found in google database, need to find a way to calculate their word vectors
