import time
import argparse
import os

from model import *
from utils import *
from options import *


def execution_time(start, end, p=False):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    if p:
        print("Execution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return int(hours), int(minutes), int(seconds)


def save_fn(model, epoch, optimizer, scheduler, loss, args, dir="../run"):
    directory = args.name if dir is None else dir + "/" + args.name
    filename = args.name + "_epoch="
    filename = filename + str(epoch) if epoch > 9 else filename + str(0) + str(epoch)
    filename = filename + "_info.pt"
    path = directory + "/" + filename
    os.makedirs(os.path.dirname(directory + "/"), exist_ok=True)
    print("Saving: ", path, "\n")
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss}, path)


def load(model, device, optimizer, scheduler, path):
    model.to(device)
    print("\n...Loading ", path, "...")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("...Loaded!")
    return {'optimizer': optimizer, 'scheduler': scheduler, 'epoch': epoch, 'loss': loss}


def train(model, train_loader, val_loader, optimizer, scheduler, loss, args, device,
          save_f, checkpoint=None):
    model.to(device)
    model.train()
    start_epoch = 0 if checkpoint is None else checkpoint['epoch'] + 1
    start = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_loss = 0.0
        problem_counter = 0
        for sat_problem in train_loader:
            optimizer.zero_grad()
            sat_problem.max_factor_dimensions = sat_problem.max_factor_dimensions[0]
            sat_problem.n_var_states = sat_problem.n_var_states[0]
            sat_problem.to(device)
            exact_ln_z = sat_problem.ln_Z
            predicted_ln_z = model(sat_problem).squeeze()
            batch_loss = loss(predicted_ln_z, exact_ln_z.float().squeeze())
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * predicted_ln_z.numel()
            epoch_loss += batch_loss.item()
            problem_counter += predicted_ln_z.numel()
        epoch_end = time.time()
        training_rmse = np.sqrt(total_loss / problem_counter)
        epoch_h, epoch_m, epoch_s = execution_time(epoch_start, epoch_end)

        print("epoch: ", epoch, "Training RMSE = ", training_rmse, " [%d:%d:%d]" % (epoch_h, epoch_m, epoch_s))
        file = open("../losses/" + args.name + "/" + "training_results_" + args.name + ".txt", "a")
        file.write("[%d] %f [%d:%d:%d]\n" % (epoch, training_rmse, epoch_h, epoch_m, epoch_s))
        file.close()

        if epoch % args.save_freq == 0:
            save_f(model, epoch, optimizer, scheduler, loss, args)

        if epoch % args.validation_freq == 0:
            val_total_loss = 0.0
            problem_counter_val = 0
            val_start = time.time()
            for sat_problem in val_loader:
                sat_problem.max_factor_dimensions = sat_problem.max_factor_dimensions[0]
                sat_problem.n_var_states = sat_problem.n_var_states[0]
                sat_problem.to(device)
                exact_ln_z = sat_problem.ln_Z
                predicted_ln_z = model(sat_problem)
                batch_loss = loss(predicted_ln_z.squeeze(), exact_ln_z.float().squeeze())
                val_total_loss += batch_loss.item() * predicted_ln_z.numel()
                problem_counter_val += predicted_ln_z.numel()
            val_end = time.time()
            val_rmse = np.sqrt(val_total_loss / problem_counter_val)
            val_h, val_m, val_s = execution_time(val_start, val_end)

            print("epoch: ", epoch, "Validation RMSE = ", val_rmse, " [%d:%d:%d]" % (val_h, val_m, val_s))
            file = open("../losses/" + args.name + "/" + "validation_results_" + args.name + ".txt", "a")
            file.write("[%d] %f [%d:%d:%d]\n" % (epoch, val_rmse, val_h, val_m, val_s))
            file.close()
        scheduler.step()
    end = time.time()
    _, _, _ = execution_time(start, end, p=True)


def test(model, test_loader, device, loss, args):
    model.to(device)
    model.eval()
    total_loss = 0.0
    problem_counter = 0
    with torch.no_grad():
        start = time.time()
        for i, sat_problem in enumerate(test_loader):
            sat_problem.max_factor_dimensions = sat_problem.max_factor_dimensions[0]
            sat_problem.n_var_states = sat_problem.n_var_states[0]
            sat_problem.to(device)
            exact_ln_z = sat_problem.ln_Z
            predicted_ln_z = model(sat_problem)
            batch_loss = loss(predicted_ln_z.squeeze(), exact_ln_z.float().squeeze())
            total_loss += batch_loss.item() * predicted_ln_z.numel()
            problem_counter += predicted_ln_z.numel()

            file = open("../results/" + args.name + "/" + args.id + ".txt", "a")
            for j in range(len(exact_ln_z)):
                file.write("%f %f\n" % (exact_ln_z[j], predicted_ln_z[j]))
            file.close()

        test_rmse = np.sqrt(total_loss / problem_counter)
        end = time.time()
        test_h, test_m, test_s = execution_time(start, end)
        print("RMSE testing = ", test_rmse)
        file = open("../losses/" + args.name + "/" + args.id + "_testing_results.txt", "a")
        file.write("%f [%d:%d:%d]\n" % (test_rmse, test_h, test_m, test_s))
        file.close()


def main(args):
    # create folders to store results and losses
    if not os.path.exists(os.path.join(os.getcwd(), "../losses/" + args.name)):
        os.makedirs("../losses/" + args.name)
    if not os.path.exists(os.path.join(os.getcwd(), "../results/" + args.name)):
        os.makedirs("../results/" + args.name)
    if args.device is not None:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    model = NeuralBP(neural_bp=args.neural_bp, max_factor_state_dimensions=args.max_factor_dimensions,
                     msg_passing_iters=args.msg_passing_iters, transform_factor_var_mlp=args.mlp_factor_var,
                     transform_var_factor_mlp=args.mlp_var_factor, transform_damp_factor_var=args.mlp_damp_factor_var,
                     attention_factor_var=args.attention_factor_var, attention_var_factor=args.attention_var_factor,
                     transform_damp_var_factor=args.mlp_damp_var_factor, learn_bethe=args.learn_bethe,
                     n_var_states=args.n_var_states, damp_param_factor_var=args.damp_param_factor_var,
                     damp_param_var_factor=args.damp_param_var_factor,
                     device=device)
    model.to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    # Loss function
    loss_fn = torch.nn.MSELoss()
    if args.train:
        print("\n...Loading training and validation data...")
        train_loader = SatFactorGraphData(args.data_dir, args.id, args.count_dir).get_data_loaders(args.batch_size,
                                                                                                   train=True)
        val_loader = SatFactorGraphData(args.data_dir, args.id, args.count_dir).get_data_loaders(args.batch_size,
                                                                                                 validation=True)
        print("...Training and validation data acquired!")
        if args.restore_train:
            if args.restore_file is None:
                directory = args.run_dir + "/" + args.name
                filenames = os.listdir(directory)
                file = directory + "/" + sorted(filenames)[-1]
            else:
                file = args.restore_file
            checkpoint = load(model, device, optimizer, scheduler, file)
            train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, args, device, save_fn,
                  checkpoint=checkpoint)
        else:
            train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, args, device, save_fn)
    if args.test:
        print("\n...Loading test data...")
        test_loader = SatFactorGraphData(args.data_dir, args.id, args.count_dir).get_data_loaders(args.batch_size,
                                                                                                  test=True)
        print("...Test data acquired!")
        print("\n Testing", args.name, "on", args.id)
        if args.train:
            test(model, test_loader, device, loss_fn, args)
        if not args.train:
            if args.restore_file is not None:
                file = args.restore_file
            else:
                directory = args.run_dir + "/" + args.name
                filenames = os.listdir(directory)
                file = directory + "/" + sorted(filenames)[-1]
            _ = load(model, device, optimizer, scheduler, file)
            test(model, test_loader, device, loss_fn, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_loader_options(parser)
    add_model_options(parser)
    add_main_options(parser)
    main(args=parser.parse_args())
