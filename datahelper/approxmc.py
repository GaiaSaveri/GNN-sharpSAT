import os
import os.path
import argparse


def approx_option(parser):
    parser.add_argument('--folder', type=str, default=None, help='directory to execute approxmc')


def approx(data_dir):
    data_fold = "../data/" + data_dir + "/test"
    count_dir = "../approx/" + data_dir
    if not os.path.exists(os.path.join(os.getcwd(), count_dir)):
        os.makedirs(count_dir)
    data_files = os.listdir(data_fold)
    for file in data_files:
        if file == "parameters.json" or file == ".DS_Store":
            continue
        else:
            count_file = file[:-7]
            path = count_dir + "/" + count_file + ".txt"
            if os.path.isfile(path):
                continue
            bash = "cd ../approxmc/approx/build; ./approxmc --epsilon=0.8 --delta=0.2 --gaussuntil=400 " + "../../" + \
                   data_fold + "/" + file + " >> " + "../../" + path
            os.system(bash)
    print("computed all: ", data_dir)


def main(args):
    approx(args.folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    approx_option(parser)
    main(args=parser.parse_args())
