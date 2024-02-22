# devcontainer.json/updateContentCommand
# executes whenever new content is available

# update apt
sudo apt update && sudo apt -y upgrade

# activate Python virtual environment
source ~/.env/bin/activate

# update required packages
python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# tell user it's done
echo '✅ Packages installed and Requirements met'
