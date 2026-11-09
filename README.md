# DEX: Demonstration-Guided RL with Efficient Exploration for Task Automation of Surgical Robot

This repository is based on the official PyTorch implementation of the paper **"Demonstration-Guided Reinforcement Learning with Efficient Exploration for Task Automation of Surgical Robot" (ICRA 2023).**

The original implementation was adapted to run on macOS with Apple Silicon and to support PPO training on the `ActiveTrack-v0` task.

## Setup and Training

### 1. Software Requirements

The code was tested on macOS with an Apple M2 Pro.

- Python 3.10.22
- PyTorch 2.14.1 with Apple Metal Performance Shaders (MPS) support
- NumPy 2.2.6
- PyBullet and SurRoL for robotic simulation

### 2. Create the Conda Environment

Create and activate a dedicated Conda environment:

```bash
conda create -n dex310 python=3.10
conda activate dex310
```

Install the project dependencies using the provided dependency files, if available. If a `requirements-mac.txt` file from the tested environment is provided, install its packages with:

```bash
python -m pip install -r requirements-mac.txt
```

**Note:** The environment must include compatible versions of PyTorch, PyBullet, SurRoL, and the other dependencies required by DEX. The original environment was tested on Apple Silicon; other operating systems may require adjustments.

### 3. Implementation and Modifications

The original DEX implementation was adapted to run on macOS with Apple Silicon (M2 Pro) and to support the `ActiveTrack-v0` task using PPO.

The following modifications and configurations were made:

- **Environment compatibility:** Adapted the original Ubuntu-based setup to work on macOS with Python 3.10 and Apple Silicon.
- **PyTorch backend:** Configured the training pipeline to support Apple's Metal Performance Shaders (MPS) backend instead of CUDA.
- **Dependency management:** Set up a dedicated Conda environment and installed the required dependencies for the DEX and SurRoL simulation environments.
- **Task configuration:** Configured the training pipeline to run the `ActiveTrack-v0` environment with PPO, which was not implemented in the original paper.

These changes allow the original DEX framework to be used for PPO training in the adapted simulation setup.

### 4. Run PPO Training

From the root directory of the DEX project, run:

```bash
python3 train.py task=ActiveTrack-v0 agent=PPO use_wb=False device=mps n_train_steps=500000
```

Configuration:

- `task=ActiveTrack-v0`: Selects the active tracking task.
- `agent=PPO`: Uses Proximal Policy Optimization.
- `use_wb=False`: Disables Weights & Biases logging.
- `device=mps`: Uses Apple's Metal GPU backend.
- `n_train_steps=500000`: Sets the training budget to 500,000 environment steps.

For systems without Apple Metal support, try `device=cpu`. For compatible NVIDIA systems, use `device=cuda` if supported by the installed code and PyTorch version.

### 5. Monitoring Training

The training logs report environment steps, completed episodes, episode length, reward, and simulation speed. Episodes may terminate early when the tracked object leaves the defined field of view. Such termination is part of the current task implementation and does not necessarily indicate a software error.

### 6. Model Evaluation

After training, the saved PPO checkpoint can be evaluated using the provided evaluation script.

To evaluate the latest checkpoint for `ActiveTrack-v0`, run:

```bash
python3 eval.py task=ActiveTrack-v0 agent=PPO ckpt_episode=latest device=mps
```

The checkpoint to be evaluated can also be specified manually using `ckpt_episode` instead of `latest`.

## Original DEX Implementation

### Prerequisites

The original implementation specifies:

- Ubuntu 18.04
- Python 3.7+

### Original Installation Instructions

1. Clone the repository:

```bash
git clone --recursive https://github.com/med-air/DEX.git
cd DEX
```

2. Create a virtual environment:

```bash
conda create -n dex python=3.8
conda activate dex
```

3. Install packages:

```bash
pip3 install -e SurRoL/
pip3 install -r requirements.txt
pip3 install -e .
```

4. Add the following line to `gym/gym/envs/__init__.py` to register SurRoL tasks:

```python
import surrol.gym
```

The file is located in the installed Gym package directory, typically:

```text
anaconda3/envs/dex/lib/python3.8/site-packages/
```

### Original Usage

The original implementation supports demonstration data collection through the scripted controllers provided by SurRoL.

For example, to collect demonstration data for the NeedlePick task:

```bash
mkdir SurRoL/surrol/data/demo
python SurRoL/surrol/data/data_generation.py --env NeedlePick-v0
```

### Original Training Commands

The original DEX repository provides training implementations for several reinforcement learning algorithms.

Train **DEX**:

```bash
python3 train.py task=NeedlePick-v0 agent=dex use_wb=True
```

Train **SAC**:

```bash
python3 train.py task=NeedlePick-v0 agent=sac use_wb=True
```

Train **DDPG**:

```bash
python3 train.py task=NeedlePick-v0 agent=ddpg use_wb=True
```

Train **DDPGBC**:

```bash
python3 train.py task=NeedlePick-v0 agent=ddpgbc use_wb=True
```

Train **CoL**:

```bash
python3 train.py task=NeedlePick-v0 agent=col use_wb=True
```

Train **AMP**:

```bash
python3 train.py task=NeedlePick-v0 agent=amp use_wb=True
```

Train **AWAC**:

```bash
python3 train.py task=NeedlePick-v0 agent=awac use_wb=True
```

Train **SQIL**:

```bash
python3 train.py task=NeedlePick-v0 agent=sqil use_wb=True
```

All commands can also be used with other surgical tasks by replacing `NeedlePick-v0` with the respective environment.

Before running the original commands with Weights & Biases logging enabled, update the WandB entity in `dex/configs/train.yaml` to match your account.

### Parallel Training

The original implementation also supports synchronous parallelization of RL training. For example, to launch four parallel training processes:

```bash
mpirun -np 4 python -m train agent=dex task=NeedlePick-v0 use_wb=True
```

Parallel training may lead to inconsistent performance and can require additional hyperparameter tuning.

### Original Evaluation Commands

The original repository also provides an evaluation script for saved models.

The directory of the model to be evaluated must be specified in `dex/configs/eval.yaml`, where the checkpoint is selected using `ckpt_episode`.

For example, to evaluate a DEX model trained on NeedlePick:

```bash
python3 eval.py task=NeedlePick-v0 agent=dex ckpt_episode=latest
```

## Starting to Modify the Code

### Modifying the Hyperparameters

The default hyperparameters are defined in `dex/configs`.

- `train.yaml` defines the experiment settings.
- The YAML files in `dex/configs/agent` define the hyperparameters of each algorithm.

Parameters can be modified directly in the experiment or agent configuration files, or passed through the terminal command.

For example:

```bash
python3 train.py task=NeedleRegrasp-v0 agent=dex use_wb=True batch_size=256 agent.aux_weight=10
```

### Adding a New RL Algorithm

The core RL algorithms are implemented within the `BaseAgent` class.

To add a new algorithm:

1. Create a new file in `dex/agents`.
2. Subclass `BaseAgent` from `dex/agents/base.py`.
3. Implement the required networks (e.g., actor and critic).
4. Override the `update(...)` and `get_action(...)` functions.
5. Register the new algorithm in `dex/agents/factory.py`.
6. Create a corresponding configuration file in `dex/configs/agent`.

For an example, see the DDPGBC implementation in `dex/agents/ddpgbc.py`.

### Transferring to Other Simulation Platforms

The code is designed for standard goal-conditioned Gym-based environments and can be transferred to other platforms that provide compatible interfaces (e.g., OpenAI Gym Fetch).

If no compatible interface is provided, modifications may be necessary, for example to the replay buffer and sampling utilities.

## Code Navigation

```text
dex
 |-- agents                # Implements core algorithms in agent classes
 |-- components            # Reusable training infrastructure
 |   |-- checkpointer.py   # Handles saving and loading model checkpoints
 |   |-- normalizer.py     # Normalizer for vectorized input
 |   |-- logger.py         # Logging functionality using WandB
 |
 |-- configs               # Experiment configurations
 |   |-- train.yaml        # Configurations for RL training
 |   |-- eval.yaml         # Configurations for RL evaluation
 |   |-- agent             # Configurations for each algorithm
 |
 |-- modules               # Reusable architecture components
 |   |-- critic.py         # Basic critic implementations
 |   |-- distributions.py  # PyTorch distribution utilities
 |   |-- policy.py         # Basic actor implementations
 |   |-- replay_buffer.py  # HER replay buffer with future sampling
 |   |-- sampler.py        # Rollout sampler for collecting experience
 |   |-- subnetworks.py    # Basic neural network components
 |
 |-- trainers              # Main training script and training loop
 |
 |-- utils                 # General RL and visualization utilities
 |-- train.py              # Experiment launcher
 |-- eval.py               # Evaluation launcher
```

## Contact

For questions regarding the original implementation, please contact: taou.cs13@gmail.com.