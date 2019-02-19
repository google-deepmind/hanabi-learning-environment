import argparse, os, re, pickle
import matplotlib.pyplot as plt

def main(args):

    # Find the latest log file in the directory and load into `data`
    log_dir = args.log_dir

    files_in_dir = os.listdir(log_dir)

    log_regex = re.compile("log_([0-9]+)")

    matches = [log_regex.match(filename) for filename in files_in_dir]
    matches = [(int(m.group(1)), m.group(0)) for m in matches if m is not None]
    matches.sort(key = lambda x : -x[0])    # Reverse sort

    latest_logfile = matches[0][1]
    with open(os.path.join(log_dir, latest_logfile), 'rb') as f:
        data = pickle.load(f)
    
    iter_keys = data.keys()

    # Will hold the iteration number and the average return on training episodes
    iterations = []
    av_returns = []

    iter_regex = re.compile("iter([0-9]+)")

    for iter_key in iter_keys:
        i = int(iter_regex.match(iter_key).group(1))
        av_return = data[iter_key]['average_return'][0]

        iterations.append(i)
        av_returns.append(av_return)

    ax = plt.axes()
    ax.plot(iterations, av_returns)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average training return")
    plt.savefig("training_performance.png")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)

    args = parser.parse_args()

    main(args)
