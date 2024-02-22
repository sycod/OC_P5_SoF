# devcontainer.json/updateContentCommand
# executes whenever new content is available

# update apt
sudo apt update && sudo apt -y upgrade

# activate Python virtual environment
source ~/.env/Scripts/activate

# update required packages
pip install --upgrade pip && pip install -r requirements.txt

# tell user it's done
echo '✅ Packages installed and Requirements met'
