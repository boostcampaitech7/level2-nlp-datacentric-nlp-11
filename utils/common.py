import pandas as pd


def split_clean_and_noise_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    df.drop(index=df[df["is_noise"] == 1.0].index, inplace=True)
    df.to_csv(
        f"only-clean-text_{file_path.split('/')[-1]}", index=False, encoding="utf-8"
    )

    df = pd.read_csv(file_path, encoding="utf-8")
    df.drop(index=df[df["is_noise"] == 0.0].index, inplace=True)
    df.to_csv(
        f"only-noise-text_{file_path.split('/')[-1]}", index=False, encoding="utf-8"
    )


if __name__ == "__main__":
    split_clean_and_noise_data(
        "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data_is-noise.csv"
    )
