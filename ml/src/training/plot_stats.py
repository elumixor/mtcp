import matplotlib.pyplot as plt


def plot_training_stats(epochs, loss_trn, loss_val, accuracy, accuracy_tth, initial_loss):
    # Plot the loss
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(epochs, loss_trn, label="Training loss")
    ax.plot(epochs, loss_val, label="Validation loss")

    # Dash line for the expected initial loss
    ax.axhline(initial_loss, c="tab:orange", ls="--", label="Initial loss")

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over epochs")

    # Plot the best val loss
    ax.axhline(min(loss_val), c="tab:green", ls="--", label="Best validation loss")

    # Plot the accuracy
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(epochs, accuracy, label="Validation accuracy")
    ax.plot(epochs, accuracy_tth, label="Validation accuracy (ttH)")

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over epochs")

    # Plot the best val accuracy
    ax.axhline(max(accuracy), c="tab:green", ls="--", label="Best validation accuracy")
