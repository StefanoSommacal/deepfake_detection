import time, os, datetime


def executeSomething():
    time.sleep(600)


def clear_results():
    results_dir = os.path.join('static', 'results')
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        for file in files:                              # delete videos older than 30 minutes
            file_path = os.path.join(results_dir, file)
            if os.path.exists(file_path) and datetime.datetime.now().timestamp() - os.path.getctime(file_path) >= 1800:  
                os.remove(file_path)
    print("Results cleared")

while True:
    clear_results()
    executeSomething()