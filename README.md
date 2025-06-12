## train some rl agent using Tianshou
### about
DQN => RainbowDQN => PPO
### quick start 
```
git clone https://github.com/shachashou/RL_TIANSHOU_EXAMPLE.git
cd RL_TIANSHOU_EXAMPLE
python3.9 -m venv .venv
.venv/Scripts/activate
pip install -r requirements_torch_cuda.txt
pip install -r requirements_other.txt
python -m TRAIN.carpole
python -m PLAY.carpole
```
