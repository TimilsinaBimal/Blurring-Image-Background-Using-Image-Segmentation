import matplotlib.pyplot as plt


def plot_loss_vs_epoch(history) -> None:
    df = history.history
    plt.plot(df.index + 1, df['loss'], label="Training Loss")
    plt.plot(df.index + 1, df['val_loss'], label="Validation Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("No. of epochs")
    plt.ylabel("Loss")
    plt.xticks(df.index + 1)
    plt.legend()
    plt.show()


def plot_accuracy_vs_epoch(history) -> None:
    df = history.history
    plt.plot(df.index + 1, df['accuracy'], label="Training Accuracy")
    plt.plot(df.index + 1, df['val_loss'], label="Validation Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("No. of epochs")
    plt.ylabel("Accuracy")
    plt.xticks(df.index + 1)
    plt.legend()
    plt.show()
