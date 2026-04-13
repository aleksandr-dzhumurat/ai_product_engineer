[![Лекция 04 vol 1: Организация кода в ML проектах](http://img.youtube.com/vi/yFGYz8XAw30/0.jpg)](http://www.youtube.com/watch?v=yFGYz8XAw30 "Лекция 04 vol 1: Организация кода в ML проектах")

start with [installing uv](https://docs.astral.sh/uv/getting-started/installation/)

How to code using agents: [learn cursor](https://cursor.com/learn)

## Creating python environment 
 
```shell
uv venv --python 3.13;
source .venv/bin/activate;
uv pip install -r requirements.txt
```

### Old way: pyenv 
Install deps

```bash
brew install openssl xz gdbm
```

Install python versions

```
pyenv install 3.12 && \
pyenv virtualenv 3.12 mlproducts-env \
source ~/.pyenv/versions/mlproducts-env/bin/activate
```

Install requirements

```shell
python3 -m pip install -r requirements.txt
```

Set up `.vscode/settings.json`

```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${workspaceFolder}/src"
    },
    "python.analysis.extraPaths": [
        "${workspaceFolder}/src"
    ],
    "editor.formatOnSave": true,
}
```

run jupyter

```shell
make run-jupyter
```

or use direct command

```shell
jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8887 \
	--NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 
```

# Data preparation


Go to [google drive](https://drive.google.com/drive/my-drive) and create directory ai_product_engineer_course 

![collab_dir_creation](img/collab_dir_creation.png)

Step 1: download data to the local machine or copy to our google drive: [ML for products](https://drive.google.com/drive/folders/1FMLKfNZZyFgzOhWjOiyeN3XvCsjT5-ET?usp=drive_link)

![datasets](img/datasets.png)

Upload jupyter_notebooks from your local machine to google drive. It is just an option, for sure you can run all jupyter code on local machine

![jupyter_notebooks_dir](img/jupyter_notebooks_dir.png)

Also upload an src dir to Google Drive.

Open first notebook [vol_00_pre_requirements_01_machine_learning_intro.ipynb](https://github.com/aleksandr-dzhumurat/ai_product_engineer/blob/main/jupyter_notebooks/vol_00_pre_requirements_01_machine_learning_intro.ipynb) and enjoy!

## Local option

Докер для винды (обе опции) https://docs.docker.com/desktop/setup/install/windows-install/

Подробнее про WSL: https://learn.microsoft.com/en-us/windows/wsl/install

## Remote option

Connect to remote server

edit `~/.ssh/config`

```python
Host remote_dev
    HostName 168.119.168.170    
    User root    
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

Connect in VSCode

- Click the green icon in the bottom-left corner
- Select "Connect to Host..."
- Choose "myserver" from the list

Install [SFTP extension](https://marketplace.visualstudio.com/items?itemName=Natizyskunk.sftp)

Install https://code.visualstudio.com/docs/devcontainers/create-dev-container

Configure with .devcontainr/devcontainer.json

```jsx
{
    "image": "python3.12",
    "runArgs": [
        "--network=host",
    ],    
    "customizations":{
        "vscode": {
            "extensions":[
                "ms-python.python",
                "ms-python.vscode-pylance"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/srv/bin/python3"
            }
        }
    }
}
```

For extension search `Ctrl`+`Shift`+`P`

Интересная опция с Cloud.ru https://cloud.ru/ - есть возможность  Удаленно подключиться к убунте и настроить докер по инструкции для убунты

Интересная ссылка: [Google collab remote](https://www.linkedin.com/posts/jeremy-arancio_struggling-with-running-llms-for-your-experimentations-activity-7376201218483863552-XIQw)

[Add ssh key to remote machine](https://community.hetzner.com/tutorials/add-ssh-key-to-your-hetzner-cloud)


## Keys generation

key generation
```shell
ssh-keygen -o  -t ed25519 -f ~/.ssh/meaningful_key_name -C "your_terminal_name"
```

Access restrictions
```shell
chmod 600 ~/.ssh/meaningful_key_name
```

Downloading repo if restricted
```shell
GIT_SSH_COMMAND="ssh -i ~/.ssh/meaningful_key_name" git clone ssh://git@gitlab.com/ORG/repo.git
```

OR modify ssh config `nano ~/.ssh/config`

```shell
Host gitlab.YOURCOMPANY.com
    HostName gitlab.YOURCOMPANY.com
    User git
    IdentityFile ~/.ssh/your_key_name
    IdentitiesOnly yes
    
Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```


# VSCode settings

- [GitHub's git cheatsheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [GitLab's git cheatsheet](https://about.gitlab.com/images/press/git-cheat-sheet.pdf)
- [learnggitbranching.js.org](learnggitbranching.js.org)
- [14-vs-code-extensions-every-data-engineer-should-swear-by-for-maximum-productivity](https://medium.com/@manojkumar.vadivel/14-vs-code-extensions-every-data-engineer-should-swear-by-for-maximum-productivity-9fcc2e1b3c4f)
- [clickhouse-support](https://marketplace.visualstudio.com/items?itemName=LinJun.clickhouse-support)
- [python-software-development-course](https://github.com/phitoduck/python-software-development-course)
- [clean-code-python](https://testdriven.io/blog/clean-code-python/)
- [refactoring](https://martinfowler.com/books/refactoring.html)
- [ohmyzsh](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH)
- [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions/blob/master/INSTALL.md#oh-my-zsh)
- [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting/blob/master/INSTALL.md)
- [ohmyz.sh](https://ohmyz.sh/#install)
- [Taking Python to Production: A Professional Onboarding Guide](https://www.notion.so/Taking-Python-to-Production-A-Professional-Onboarding-Guide-799409731bf14c78a531ac779f1bd76d?pvs=21) 
- [keyboard-shortcuts-macos](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf)
- [keyboard-shortcuts-linux](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf)

# Bonus: setting up remote server for multi users

# SSH & User Management on Alibaba Cloud (Ubuntu)

## 1. Add SSH Public Key to Remote Machine

Three methods to allow key-based login:

- **`ssh-copy-id`** (easiest): `ssh-copy-id -i ~/.ssh/id_rsa.pub user@server-ip`
- **Manual**: Append key to `~/.ssh/authorized_keys`, set permissions `chmod 600`
- **One-liner pipe**: `cat ~/.ssh/id_rsa.pub | ssh user@server-ip "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"`

Key permissions: `~/.ssh` must be `700`, `authorized_keys` must be `600`.

---

## 2. Create User `trova` with Password Login

```bash
sudo adduser trova          # creates user + home dir, prompts for password
```

Enable password authentication in `/etc/ssh/sshd_config`:
```
PasswordAuthentication yes
PermitEmptyPasswords no
```

Restart SSH:
```bash
sudo systemctl restart sshd
```

Connect from local:
```bash
ssh trova@your-server-ip
```

Optional — grant sudo: `sudo usermod -aG sudo trova`

---

## 3. Generated Passwords

| # | Password |
|---|---|
| 1 | `Kx9#mP2$vL4nQw7!` |
| 2 | `Rj5@tN8&hY3bZe6*` |

> Store in a password manager, not plain text.

---

## 4. Remove User or Disable Login

| Goal | Command |
|---|---|
| Delete user + home dir | `sudo userdel -r trova` |
| Lock account | `sudo usermod -L trova` |
| Block shell access | `sudo usermod -s /sbin/nologin trova` |
| Expire account immediately | `sudo usermod --expiredate 1 trova` |
| Remove SSH keys | `sudo rm /home/trova/.ssh/authorized_keys` |
| Kill active session | `sudo pkill -u trova` |

Verify user is gone: `id trova`