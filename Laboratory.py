from IWOA import BaoWOA
from utils.FunctionUtil import whale_f1, whale_f2, whale_f3,  whale_f5, whale_f6, whale_f7, whale_f8, whale_f9, whale_f10, whale_f11
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gspread
from oauth2client.service_account import ServiceAccountCredentials



## Setting parameters
root_paras = {
    "problem_size": 2,
    "domain_range": [-10,10],
    "print_train": True,
    "objective_func": whale_f3
}
woa_paras = {
    "epoch": 20,
    "pop_size": 10
}


benchmark_functions = [
    {"func": whale_f1, "name": "Sphere function", "domain_range": (-100, 100), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f2, "name": "Sum of absolute values", "domain_range": (-10, 10), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f3, "name": "Sum of squares function", "domain_range": (-10, 10), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f5, "name": "Rosenbrock", "domain_range": (-2.048, 2.048), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f6, "name": "Step function", "domain_range": (-100, 100), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f7, "name": "Quartic function (w/ noise)", "domain_range": (-1.28, 1.28), "epoch": 500, "problem_size": 9, "pop_size": 10},
    {"func": whale_f8, "name": "Schwefel", "domain_range": (-500, 500), "epoch": 500, "problem_size": 1, "pop_size": 10},
    {"func": whale_f9, "name": "Rastrigin", "domain_range": (-5.12, 5.12), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f10, "name": "Ackley", "domain_range": (-32, 32), "epoch": 500, "problem_size": 2, "pop_size": 10},
    {"func": whale_f11, "name": "Griewank", "domain_range": (-600, 600), "epoch": 500, "problem_size": 2, "pop_size": 10}
]

## Run model
#md = BaoWOA(root_algo_paras=root_paras, woa_paras=woa_paras)
#md._train__()


def run_and_plot_experiments(benchmark_functions):
    results = []

    for bf in benchmark_functions:
        root_paras = {"problem_size": bf["problem_size"],
                       "domain_range": bf["domain_range"],
                         "objective_func": bf["func"]}
        
        woa_paras = {"epoch": bf["epoch"],
                      "pop_size": bf["pop_size"]}
        
        
        optimizer = BaoWOA(root_algo_paras=root_paras, woa_paras=woa_paras)
        mean_fitness, std_dev_fitness = optimizer.run_and_collect_stats()  # This method needs to be implemented
        mean_fitness_rounded = round(mean_fitness, 15)  
        std_dev_fitness_rounded = round(std_dev_fitness, 15)
        results.append({"function name": bf["name"], "epochs": bf["epoch"], "problem size": bf["problem_size"], "mean of last fitness": mean_fitness_rounded, "standard deviation of the last fitness": std_dev_fitness_rounded})

    return pd.DataFrame(results)



df_results = run_and_plot_experiments(benchmark_functions)

# Print the results
df_results['mean of last fitness'] = df_results['mean of last fitness']
df_results['standard deviation of the last fitness'] = df_results['standard deviation of the last fitness']

print(df_results)

# Save the results as csv
csv_file_path = 'D:/Oluwasegun/Desktop/MSc/Winter Semester/Thesis/Optimization Scripts/Thesis_Laboratory/Test results/log_experiment_results.csv'
df_results.to_csv(csv_file_path, index=False)


#mean_fitness, std_dev_fitness = run_and_plot_experiments(root_paras, woa_paras, num_experiments=10)

#optimizer = BaoWOA(root_algo_paras=root_algo_paras, woa_paras=woa_paras)
#mean_fitness, std_dev_fitness = optimizer.run_and_collect_stats(num_experiments=10)




'''
def run_and_plot_experiments(root_algo_paras, woa_paras, num_experiments=10):
    all_experiments_results = []
    last_epoch_fitness_values = []  # List to store the last epoch's fitness value of each experiment

    
    for exp in range(num_experiments):
        optimizer = BaoWOA(root_algo_paras=root_algo_paras, woa_paras=woa_paras)
        _, experiment_results = optimizer._train__()
        all_experiments_results.append(experiment_results)
        last_epoch_fitness_values.append(experiment_results[-1])
    
    # Calculate the mean and standard deviation of the last fitness values
    mean_fitness = np.mean(last_epoch_fitness_values)
    std_dev_fitness = np.std(last_epoch_fitness_values)
    
    print(f'Mean of the last fitness values: {mean_fitness}')
    print(f'Standard deviation of the last fitness values: {std_dev_fitness}')

    plt.figure(figsize=(10, 6))
    for i, results in enumerate(all_experiments_results):
        plt.plot(results, label=f'Experiment {i+1}')

    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.show()

    return mean_fitness, std_dev_fitness
'''







# Load results into google sheets
'''
 Define the scope  
scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']

# Add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name(
    'D:/Oluwasegun/Desktop/MSc/Winter Semester/Thesis/Optimization Scripts/Thesis_Laboratory/client_secret_64373508783-dv3ir7vin93kb3vpv9m2kqq64kreupei.apps.googleusercontent.com.json', scope)

 Authorize the clientsheet 
client = gspread.authorize(creds)

 #Open the sheet
sheet = client.open('Thesis_experiment_stats').sheet1  # Change 'Your Google Sheet Name' to your actual sheet name


mean = mean_fitness  # Replace with your actual mean
std_dev = std_dev_fitness  # Replace with your actual standard deviation
data = [str(root_paras['objective_func']), '', woa_paras['epoch'], woa_paras['pop_size'], mean, std_dev]

sheet.append_row(data)
'''






























