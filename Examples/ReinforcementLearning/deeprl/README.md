
Examples of running CNTK DeepRL toolkit.

Dependency:

    - OpenAI Gym: https://gym.openai.com/docs
    
    - Atari: https://github.com/openai/gym#atari
             Use the following command to install Atari games on Windows:
                pip install git+https://github.com/Kojoley/atari-py.git

The following commands assume Examples/ReinforcementLearning/deeprl/scripts as the working directory.

To train an agent using

    - TabularQLearning
    python run.py --env=CartPole-v0 --max_steps=100000 --agent_config=config_examples/tabular_qlearning.config --eval_period=1000 --eval_steps=20000

    - QLearning
    python run.py --env=CartPole-v0 --max_steps=100000 --agent_config=config_examples/qlearning.config --eval_period=1000 --eval_steps=20000

    - ActorCritic
    python run.py --env=CartPole-v0 --max_steps=100000 --agent_config=config_examples/policy_gradient.config --eval_period=1000 --eval_steps=20000

    - RandomAgent
    python run.py --env=CartPole-v0 --max_steps=100 --eval_period=1 --eval_steps=200000

Use QLearning as an example, the command
```bash
python run.py --env=CartPole-v0 --max_steps=100000 --agent_config=config_examples/qlearning.config --eval_period=1000 --eval_steps=20000
```
tells QLearning agent to interact with environment CartPole-v0 for a maximum of
100000 steps, while evaluation is done every 1000 steps. Each evaluation reports
average reward per episode by interacting with the environment 20000 steps.

The agent configs, best model and evaluation results are written to --output_dir,
which defaults to 'output' in the working directory. To view the evaluation
results, type the following command in python:

```python
import shelve
d = shelve.open('output/output.wks')
d['reward_history']
d.close()
```

Note, reading and writing wks simultaneously will corrupt the file. To
check your results while the program is still running, make a copy of wks file
and read the numbers from the copy.
