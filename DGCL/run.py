
import subprocess

classification_tasks = ['bbbp', 'bace', 'sider', 'tox21', 'hiv', 'clintox']
regression_tasks = ['esol', 'freesolv', 'lipo']
random_seeds = [0, 1, 2 ,3]

def run_script(task, script_name, seed):
    print(f"Running {script_name} for task {task} with random seed {seed}")
    subprocess.run(["python", script_name, "--task", task, "--random_seed", str(seed)])

for task in classification_tasks + regression_tasks:
    script_name = "main_gnn_classification.py" if task in classification_tasks else "main_gnn_regression.py"
    for seed in random_seeds:
        run_script(task, script_name, seed)

