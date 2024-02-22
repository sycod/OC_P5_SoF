# .devcontainer/onCreateCommand.sh
# apply once at the beginning of VM creation

# update apt
sudo apt update && sudo apt -y upgrade

# create Python virtual environment
python -m venv ~/.env
