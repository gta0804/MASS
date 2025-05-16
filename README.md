# MASS: Multi-Agent Simulation Scaling  for Portfolio Construction

📜 [Paper Link](https://arxiv.org/abs/2505.10278)
## 🧑‍💻Usage
1. **dependency installation**
```
conda create -n your_env_name python==3.10 -y
conda activate your_env_name
pip install pdm
pdm install
```
2. **dataset fetching**
The dataset is under review now. We will release our dataset once the review is finalized.
After fetching dataset, change all `ROOT_PATH` variables to your dataset directory.

3. **Compute resources configuration.**
We use [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) as our foundation model. You can change your foundation model url [here](https://github.com/gta0804/MASS/blob/main/stock_disagreement/agent/basic_agent.py#L57).
For SSE 50 and the default configuration, 80GiB RAM is needed. You can save memory overhead by adjusting the agent parallelism [here](https://github.com/gta0804/MASS/blob/main/stock_disagreement/exp/trainer.py#L148).

4. **Running MASS**
```
python stock_disagreement/main.py
```

