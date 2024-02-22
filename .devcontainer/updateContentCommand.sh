# devcontainer.json/updateContentCommand
sudo apt update && sudo apt -y upgrade
pip install --upgrade pip && pip install -r requirements.txt
echo '✅ Packages installed and Requirements met'