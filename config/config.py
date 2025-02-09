class Config(object):
    torch_seed = 17
    project_name = "arcface-pytorch"  # wandb project name
    run_name = "arcface-pytorch-default"  # wandb run name

    env = "default"
    backbone = "resnet18"
    classify = "softmax"
    num_classes = 43
    metric = "arc_margin"
    easy_margin = False
    use_se = False
    loss = "focal_loss"

    display = False
    finetune = False

    train_root = "../arcface-tf2/data/graph_data_augmented"

    test_root = "/data1/Datasets/anti-spoofing/test/data_align_256"
    test_list = "test.txt"

    lfw_root = "/data/Datasets/lfw/lfw-align-128"
    lfw_test_list = "/data/Datasets/lfw/lfw_test_pair.txt"

    checkpoints_path = "checkpoints"
    load_model_path = "models/resnet18.pth"
    test_model_path = "checkpoints/resnet18_110.pth"
    save_interval = 10

    train_batch_size = 20  # batch size
    test_batch_size = 8

    input_shape = (1, 256, 256)

    optimizer = "sgd"

    use_gpu = True  # use GPU or not
    gpu_id = "0, 1"
    num_workers = 8  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = "/tmp/debug"  # if os.path.exists(debug_file): enter ipdb
    result_file = "result.csv"

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-3
