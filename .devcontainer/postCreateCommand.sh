# .devcontainer/postCreateCommand.sh
# apply once at the end of VM creation

# launch Python virtual environment on startup
echo "source ~/.env/Scripts/activate" >> ~/.bashrc

# alias
echo "alias ll='ls -alF'" >> ~/.bashrc
