# similarity
find the most similar user
## useage
run in command line:

python main.py userkey1 uesrkey2 userkey3 (optional: --k=somenumber --model_path=path to model file --user_data_path = path to user data --patch_path = path to patch file --verbosity = True or False, show progress)

example:

python main.py 23456 56447 --k=2000

if update file is provided, run as:

python main.py 23456 --patch_path=update.json

and it will give a list of top N (default 3) users that are most similar to the user

also it can find similar tags:
if run in command line:

python tag_matcher.py tag1 tag2 (optional: --topn=default 5 --model_path=path to model file --user_data_path = path to user data --patch_path = path to patch file --verbosity = True or False, show progress)

example:

python tag_matcher.py maths physics --topn=

matching design group:

run in command line:

python running_script_for_design_group.py

##warning
running this code first time is very very slow because it needs to create all the required model files, once those files are created, the code will be much faster. 
## todo
Output should be randomized a little bit, so that it does not give same outcome every time.

Some tags cannot be found in google database, need to find a way to calculate their word vectors
