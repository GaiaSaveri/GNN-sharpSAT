def add_loader_options(parser):
    parser.add_argument('--data_dir', default='../data', type=str, help="Data directory")
    parser.add_argument('--count_dir', default='../counts', type=str, help="Counts directory")
    parser.add_argument('--id', default="test", type=str, help="Data/counts subdirectory")
    parser.add_argument('--max_factor_dimensions', default=5, type=int, help="Max number of vars for clause")
    parser.add_argument('--batch_size', default=10, type=int, help="Number of formulae in each batch")


def add_model_options(parser):
    parser.add_argument('--msg_passing_iters', type=int, default=5, help="Number of BPNN iters")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument('--step_size', type=int, default=200, help="Step size for the scheduler")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="Learning rate decay for the scheduler")
    parser.add_argument('--damp_param_factor_var', type=float, default=0.5, help="Damping factor-to-variable messages")
    parser.add_argument('--damp_param_var_factor', type=float, default=0.5, help="Damping variable-to-factor messages")
    parser.add_argument('--neural_bp', type=eval, default=True,
                        help="Whether to perform Neural Loopy Belief Propagation or standard Loopy Belief Propagation")
    parser.add_argument('--mlp_factor_var', type=eval, default=False, help="MLP applied to factor-to-var messages")
    parser.add_argument('--mlp_var_factor', type=eval, default=False, help="MLP applied to var-to-factor messages")
    parser.add_argument('--attention_factor_var', type=eval, default=False,
                        help="Transform factor-to-var messages with GAT")
    parser.add_argument('--attention_var_factor', type=eval, default=False,
                        help="Transform var-to-factor messages with GAT")
    parser.add_argument('--mlp_damp_factor_var', type=eval, default=False,
                        help="Learn damping for factor-to-var messages")
    parser.add_argument('--mlp_damp_var_factor', type=eval, default=False,
                        help="Learn damping for var-to-factor messages")
    parser.add_argument('--learn_bethe', type=eval, default=True,
                        help="If 'none' then use the standard bethe approximation with no learning otherwise a MLP")
    parser.add_argument('--n_var_states', type=int, default=2, help="Number of states each variable can take")


def add_main_options(parser):
    parser.add_argument('--name', default="Exp", type=str, help="Name of the model")
    parser.add_argument('--n_epochs', type=int, default=1001, help="Number of epochs")
    parser.add_argument('--run_dir', type=str, default="../run",
                        help="Name of the directory in which checkpoints are saved")
    parser.add_argument("--train", default=True, type=eval, help="True if training is required")
    parser.add_argument("--restore_train", default=False, type=eval, help="True if training needs to be continued")
    parser.add_argument("--test", default=True, type=eval, help="True if testing is required")
    parser.add_argument('--restore_file', default=None, type=str, help="Checkpoint to restore")
    parser.add_argument('--validation_freq', default=100, type=int, help="Frequency (in epochs) of validation step")
    parser.add_argument('--save_freq', default=200, type=int, help="Frequency (in epochs) of saving checkpoints")
    parser.add_argument('--data_name', default='test', type=str, help="Name of the dataset used for testing")
    parser.add_argument('--device', default=None, type=str, help="Device (GPU/CPU) use to run the code")
