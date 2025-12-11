# Download the .gml file
BASE_DOTE_DIR="/home/ubuntu/MetaEase/MetaOptimize/src/DOTE"
cd $BASE_DOTE_DIR
cd networking_envs
cd data

python gml_to_dote.py
# /home/ubuntu/MetaEase/MetaOptimize/src/DOTE/networking_envs/data/Abilene must be created

# Now we should compute the optimal flow for each demand matrix
cd $BASE_DOTE_DIR/networking_envs/data/Abilene
python ../compute_opts.py

cd $BASE_DOTE_DIR/networking_envs
python ../train_and_evaluate.py Abilene