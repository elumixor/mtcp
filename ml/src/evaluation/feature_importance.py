import torch
from tqdm import tqdm

from src.data import Data
from src.nn import Model


def feature_importance(model: Model,
                       data: Data,
                       class_idx: int,
                       feature_names: list[str],
                       num_steps=100,
                       num_examples=None,
                       plot=False,
                       device="cpu",
                       return_fig=False):
    model.eval()
    model.to(device)

    if num_examples is None:
        num_examples = data.n_samples

    # Get the baseline
    baseline = data[:1]
    baseline = Data.zeros(data.n_features_continuous, data.categorical_sizes, **data.metadata).to(device)

    igs = []
    indices = torch.randint(0, data.n_samples, (num_examples,))
    for i in tqdm(indices, desc="Calculating integrated gradients"):
        batch = data[i:i+1].to(device)

        # Get the difference
        x_diff_continuous = batch.x_continuous - baseline.x_continuous
        # x_diff_categorical = batch.x_categorical - baseline.x_categorical

        grads_continuous = []
        grads_categorical = []
        for alpha in torch.linspace(0, 1, num_steps):
            x_step_continuous = baseline.x_continuous + alpha * x_diff_continuous
            x_step_categorical = torch.where(alpha < 0.5, baseline.x_categorical, batch.x_categorical) # I can't think of anything better atm

            x_step_continuous.requires_grad = True
            # x_step_categorical.requires_grad = True

            step = Data(x_step_continuous, x_step_categorical, batch.y, batch.w, **batch.metadata).to(device)

            probs = model(step).softmax(dim=-1)[0, class_idx]

            # Calculate the gradient
            probs.backward()

            # Get the gradients and process them
            # Replace None with zeros and detach them from the graph
            grad_continuous = step.x_continuous.grad.view(-1).detach()
            # grad_categorical = step.x_categorical.grad.view(-1).detach()

            # if grad is None:
            #     grad = torch.zeros_like(step.features)
            # else:
            #     grad = grad.detach()

            # Finally, save the computed gradients
            grads_continuous.append(grad_continuous)
            # grads_categorical.append(grad_categorical)

        grads_continuous = torch.stack(grads_continuous)  # (num_steps, num_features)
        # grads_categorical = torch.stack(grads_categorical)  # (num_steps, num_features)

        # Use the trapezoidal rule to calculate the integral
        grads = grad_continuous
        grads = (grads[:-1] + grads[1:]) / 2  # (num_steps - 1, num_features)
        grads = grads.mean(axis=0)  # (num_features,)

        ig = grads * x_diff_continuous.view(-1)
        ig = ig.nan_to_num(nan=0)

        igs.append(ig)

    # Calculate the mean value of the integrated gradients over the whole dataset
    ig = torch.stack(igs).mean(axis=0)

    fig = None
    if plot == True or plot == "vertical":
        import matplotlib.pyplot as plt

        # Now we can plot the integrated gradients
        # For now let's drop the object feature (our simple model doesn't use it anyway)
        fig = plt.figure(figsize=(15, 25))
        ig_sorted, indices = torch.sort(ig, descending=True)
        plt.barh(torch.arange(len(ig_sorted)), ig_sorted.cpu())
        plt.yticks(torch.arange(len(ig_sorted)), [feature_names[i] for i in indices])
    elif plot == "horizontal":
        import matplotlib.pyplot as plt

        # Now we can plot the integrated gradients
        # For now let's drop the object feature (our simple model doesn't use it anyway)
        fig = plt.figure(figsize=(20, 10))
        ig_sorted, indices = torch.sort(ig, descending=True)
        plt.bar(torch.arange(len(ig_sorted)), ig_sorted.cpu())
        plt.xticks(torch.arange(len(ig_sorted)), [feature_names[i] for i in indices], rotation=90)

    if return_fig:
        return ig, fig

    return ig
