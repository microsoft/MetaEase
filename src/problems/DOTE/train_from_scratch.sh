# Download the .gml file
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DOTE_DIR="$SCRIPT_DIR"
cd $BASE_DOTE_DIR
cd networking_envs
cd data

python gml_to_dote.py
# $BASE_DOTE_DIR/networking_envs/data/Abilene must be created

# Now we should compute the optimal flow for each demand matrix
cd $BASE_DOTE_DIR/networking_envs/data/Abilene
python ../compute_opts.py

cd $BASE_DOTE_DIR/networking_envs
python ../train_and_evaluate.py Abilene