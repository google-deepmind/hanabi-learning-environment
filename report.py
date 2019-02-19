import argparse, os, re, pickle, subprocess
import matplotlib.pyplot as plt

def main(args):

    # Get the diff relative to the latest commit (if any)
    diff = subprocess.run(['git', 'diff', 'HEAD'], stdout=subprocess.PIPE)
    hash_git = subprocess.run(['git', 'log', '--oneline', '-n 1', '--pretty=%H'], stdout=subprocess.PIPE)
    short_hash = subprocess.run(['git', 'log', '--oneline', '-n 1', '--pretty=%h'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

    # Test if the parent output directory exists
    parent_output_dir = "./performance_logs"
    if (not os.path.isdir(parent_output_dir)):
        raise IOError("{} does not exist or is not a directory".format(parent_output_dir))
    # Get a suitable directory name for output
    files_in_dir = os.listdir(parent_output_dir)
    if files_in_dir.count(short_hash) == 0:
        output_dirname = os.path.join(parent_output_dir, short_hash)
    else:
       hash_filename_regex = re.compile("{}_([0-9]+)".format(short_hash))
       matches = [hash_filename_regex.match(f) for f in files_in_dir]
       matches = [int(m.group(1)) for m in matches if m is not None]
       if matches == []:
           max_n = 0
       else:
           matches.sort()
           max_n = matches[-1]
       output_dirname = os.path.join(parent_output_dir, "{}_{}".format(short_hash, max_n+1))
    os.mkdir(output_dirname)


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

    # Plot the performance
    ax = plt.axes()
    ax.plot(iterations, av_returns)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average training return")
    plt.savefig(os.path.join(output_dirname, "training_reward_vs_iteration.png"))

    # Get the average performance over the last 100 iterations
    av_last_100 = (float(sum(av_returns[-100:])))/len(av_returns[-100:])

    with open(os.path.join(output_dirname, "./report.md"), "wt") as log_markdown:
        log_markdown.write("Performance report\n")
        log_markdown.write("==================\n")
        log_markdown.write("\n\n")

        log_markdown.write("Commit `{}`.\n\n".format(hash_git.stdout.decode('utf-8').strip()))

        log_markdown.write("# Performance\n\n")
        log_markdown.write("Average loss of last 100 training iterations: {}\n\n".format(av_last_100))
        log_markdown.write("![training_reward_vs_iteration](./training_reward_vs_iteration.png)\n\n")

        log_markdown.write("# Diff\n\n")
        log_markdown.write("```\n");
        log_markdown.write(diff.stdout.decode('utf-8'))
        log_markdown.write("```\n");

    # Append onto the page linking to the individual reports
    with open(os.path.join(parent_output_dir, "README.md"), 'at') as f:
        f.write("| [{}](../{}/report.md) | {} | {} |\n".format(short_hash, output_dirname, av_last_100, iterations[-1]))

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)

    args = parser.parse_args()

    main(args)
