import os
import os.path
import argparse
from sat_gen import SatProblems


def options(parser):
    parser.add_argument('--data_dir', default='../data', type=str, help="Parent Directory for the dataset")
    parser.add_argument('--id', type=str, help="Actual directory of the dataset")
    parser.add_argument('--count_dir', default="../counts", type=str, help="Parent directory for the counts")
    parser.add_argument('--n_problems', default=1000, type=int, help="Number of CNF problems in the dataset")
    parser.add_argument('--max_cl', type=int, help="Max number of clauses in the CNF formula")
    parser.add_argument('--min_cl', type=int, help="Min number of clauses in the CNF formula")
    parser.add_argument('--max_var', type=int, help="Max number of vars in the CNF formula")
    parser.add_argument('--min_var', type=int, help="Min number of vars in the CNF formula")
    parser.add_argument('--train', default=True, type=eval, help="Whether to generate train/validation data")
    parser.add_argument('--test', default=True, type=eval, help="Whether to generate test data")


class SatCount(SatProblems):
    def __init__(self, data_dir: str, id: str, count_dir: str, n_problems: int, max_cl: int, min_cl: int, max_var: int,
                 min_var: int, kind: str):
        super().__init__(data_dir, id, n_problems, max_cl, min_cl, max_var, min_var)
        self.count_dir = count_dir
        self.kind = kind
        # generate data, or check existence
        if self.kind == "train":
            self.prepare_data(train=True)
        elif self.kind == "test":
            self.prepare_data(test=True)
        else:
            assert ("validation" == self.kind), self.kind
            self.prepare_data(validation=True)

    def count(self):
        count_dir = os.path.join(self.count_dir, self.id, self.kind)
        if not os.path.exists(os.path.join(os.getcwd(), count_dir)):
            os.makedirs(count_dir)
        data_dir = os.path.join(self.data_dir, self.id, self.kind)
        data_files = os.listdir(data_dir)
        for file in data_files:
            if file == "parameters.json" or file == ".DS_Store":
                continue
            else:
                count_file = file[:-7]
                path = count_dir + "/" + count_file + ".txt"
                if os.path.isfile(path):
                    continue
                bash = "cd ../sharpSAT/build/Release; ./sharpsat " + "../../" + data_dir + "/" + file + \
                       " | tail -1 >> " + "../../" + path
                os.system(bash)


def main(args):
    generator = SatProblems(args.data_dir, args.id, args.n_problems, args.max_cl, args.min_cl, args.max_var,
                            args.min_var)
    if args.train:
        generator.prepare_data(train=True)
        generator.prepare_data(validation=True)
    if args.test:
        generator.prepare_data(test=True)
    print("Data have been generated")
    if args.train:
        counter_train = SatCount(args.data_dir, args.id, args.count_dir, args.n_problems, args.max_cl, args.min_cl,
                                 args.max_var, args.min_var, kind="train")
        counter_train.count()
        counter_val = SatCount(args.data_dir, args.id, args.count_dir, args.n_problems, args.max_cl, args.min_cl,
                               args.max_var, args.min_var, kind="validation")
        counter_val.count()
    if args.test:
        counter_test = SatCount(args.data_dir, args.id, args.count_dir, args.n_problems, args.max_cl, args.min_cl,
                                args.max_var, args.min_var, kind="test")
        counter_test.count()
    print("Counts have been generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)
    main(args=parser.parse_args())
