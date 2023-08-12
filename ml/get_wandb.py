import wandb


def get_wandb(trn, val, model, config, tags=None):
    wandb_config = dict(
        trn_size=trn.n_samples,
        val_size=val.n_samples,
        n_classes=trn.n_classes,
        n_parameters=model.n_params,
        **config,
    )

    del wandb_config["run_name"]

    tags = config.tags + (tags if tags is not None else [])
    wandb_run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=wandb_config,
        tags=tags,
    )
    print()

    # Define min/max metrics for W&B
    wandb.define_metric("epoch")

    wandb.define_metric("trn/loss", step_metric="epoch", summary="min")
    wandb.define_metric("val/loss", step_metric="epoch", summary="min")

    wandb.define_metric("val/acc/bin", step_metric="epoch", summary="max")
    wandb.define_metric("val/acc/multi", step_metric="epoch", summary="max")

    wandb.define_metric("val/f1", step_metric="epoch", summary="max")

    wandb.define_metric("val/auc_w/ttH", step_metric="epoch", summary="max")
    wandb.define_metric("val/auc_w/mean", step_metric="epoch", summary="max")
    wandb.define_metric("sig/significance", step_metric="epoch", summary="max")

    return wandb_run
