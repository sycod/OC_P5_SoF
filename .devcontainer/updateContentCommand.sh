# devcontainer.json/updateContentCommand
# executes whenever new content is available

# update apt
sudo apt update && sudo apt -y upgrade

# activate Python virtual environment, update pip and install packages
source ~/.env/bin/activate &&
which python && which pip &&
echo '✅ Python virtual environment activated' &&
python -m pip install --upgrade pip &&
python -m pip install -r requirements.txt &&
echo '✅ Packages installed and Requirements met'
