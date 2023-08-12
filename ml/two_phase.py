import autorootcwd  # Do not delete - adds the root of the project to the path

import torch

from ml.evaluation import evaluate
from ml.get_wandb import get_wandb
from ml.training import find_lr, train, load_checkpoint

# Step 1. Get the model to be fine-tuned
if __name__ == "__main__":
    import argparse
    from ml.download_model import download_model, get_run
    from ml.nn import ResNet, Transformer
    from ml.utils import get_config
    from ml.data import load_from_config

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--clean", action="store_true", help="Clean the model directory")
    args = parser.parse_args()

    # Read the config
    model_path, checkpoint = download_model(args.config, clean=args.clean, checkpoint=True, silent=True)

    config = get_config(args.config, check_cuda_device=False, silent=True)

    # Get the start epoch from the wandb api
    run = get_run(config.run_name)
    max_auc = float("-inf")
    start_epoch = 0
    history = run.history(keys=["val/auc_w/ttH", "epoch"])
    for epoch, auc in zip(history["epoch"], history["val/auc_w/ttH"]):
        if auc > max_auc:
            max_auc = auc
            start_epoch = epoch
    print(f"Start epoch: {start_epoch}")

    # Load the model
    model_class = ResNet if config.model == "resnet" else Transformer

    def get_model():
        model = model_class.from_saved(model_path,
                                       device=config.device,
                                       #    compile=True,
                                       compile=False,
                                       return_stats=False)

        model.use_binary = True
        model.signal_class_idx = 0

        model = torch.compile(model)

        return model

    # Load the validation set
    trn, val, tst = load_from_config(config)

    F = 1 / (1 - config.trn_split)

    # Optimizer definition
    def Optim(params, lr):
        return torch.optim.AdamW(params, lr=lr)

    scheduler = None

    model = get_model()

    config.run_name = f"{config.run_name}-two-fase"
    wandb_run = get_wandb(trn, val, model, config, tags=["two-fase"])

    lr, min_loss = find_lr(
        model,
        trn,
        Optim,
        batch_size=config.batch_size,
        lr_divisions=100,
        device=config.device,
        run=wandb_run,
        half=config.dtype,
    )

    print(f"Found lr={lr:.8f} with min_loss={min_loss:.8f}")
    print()

    model = get_model()
    optim = Optim(model.parameters(), lr)

    evaluator = evaluate.using(
        model,
        trn,
        val,
        "ttH",
        device=config.device,
        use_tqdm=True,
        F=F,
        batch_size=config.batch_size,
        half=config.dtype,
    )

    stats_last, *_ = train(
        model,
        optim,
        trn,
        evaluator,
        epochs=config.epochs,
        validate_freq=1,
        restart=True,
        use_tqdm=True,
        device=config.device,
        checkpoints_dir=config.checkpoints_dir,
        batch_size=config.batch_size,
        scheduler=scheduler,
        half=config.dtype,
        start_epoch=start_epoch,
        run=wandb_run,
    )
