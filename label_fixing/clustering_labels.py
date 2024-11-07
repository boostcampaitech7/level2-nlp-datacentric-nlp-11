from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns


class LabelCluster:

    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    def embedding(self, file_path):
        self.model.eval()
        data = pd.read_csv(file_path)

        clean_embed_list, clean_label_list = [], []
        noise_embed_list, noise_label_list = [], []

        for idx, sample in tqdm(data.iterrows()):
            # is_noise는 텍스트 노이즈 여부를 의미한다. 텍스트 노이즈가 있을 경우 라벨 노이즈는 없다.
            if sample["is_noise"] == 1.0:
                input_ids = self.tokenizer(sample["text"], return_tensors="pt")[
                    "input_ids"
                ].to(self.device)

                with torch.no_grad():
                    clean_embed_list.append(
                        self.model(input_ids).pooler_output.cpu().numpy()
                    )
                    clean_label_list.append(sample["target"])
            else:
                input_ids = self.tokenizer(sample["text"], return_tensors="pt")[
                    "input_ids"
                ].to(self.device)

                with torch.no_grad():
                    noise_embed_list.append(
                        self.model(input_ids).pooler_output.cpu().numpy()
                    )
                    noise_label_list.append(sample["target"])

        return clean_embed_list, clean_label_list, noise_embed_list, noise_label_list

    def clustering(
        self,
        clean_embed_list,
        clean_label_list,
        noise_embed_list,
        noise_label_list,
        n_components=2,
    ):
        t_sne = TSNE(n_components=n_components)
        all_embed_reduced = t_sne.fit_transform(
            np.array(clean_embed_list + noise_embed_list).squeeze()
        )
        clean_embed_reduced, noise_embed_reduced = (
            all_embed_reduced[: len(clean_embed_list), :],
            all_embed_reduced[len(clean_embed_list) :, :],
        )

        palette = sns.color_palette("bright", 10)
        sns.scatterplot(
            x=clean_embed_reduced[:, 0],
            y=clean_embed_reduced[:, 1],
            hue=clean_label_list,
            legend="full",
            palette=palette,
        )
        plt.savefig("T-SNE.png", format="png")
        plt.close()

        gmm = GaussianMixture(n_components=7, random_state=42, max_iter=20)
        gmm.fit(all_embed_reduced)
        restore_label = gmm.predict(noise_embed_reduced)

        sns.scatterplot(
            x=noise_embed_reduced[:, 0],
            y=noise_embed_reduced[:, 1],
            hue=restore_label,
            legend="full",
            palette=palette,
        )
        plt.savefig("GMM.png", format="png")
        plt.close()

        return restore_label

    def restore_label(self, file_path):
        clean_embed_list, clean_label_list, noise_embed_list, noise_label_list = (
            self.embedding(file_path)
        )
        restore_label = self.clustering(
            clean_embed_list, clean_label_list, noise_embed_list, noise_label_list
        )

        df = pd.read_csv(file_path)
        df.drop(index=df[df["is_noise"] == 1.0].index, inplace=True)

        df["target"] = restore_label
        df["ID"] = ["cluster-" + id for id in df["ID"]]

        df.to_csv(
            f"cluster-label_{file_path.split('/')[-1]}", index=False, encoding="utf-8"
        )


def mapping_label(file_path, map):
    df = pd.read_csv(file_path)
    for idx in range(len(df["ID"])):
        df["target"][idx] = map[df["target"][idx]]

    df.to_csv(f"change-label_{file_path.split('/')[-1]}", index=False, encoding="utf-8")


if __name__ == "__main__":
    """
    labelcluster = LabelCluster("/data/ephemeral/home/gj/output/checkpoint-210")
    labelcluster.restore_label("/data/ephemeral/home/gj/bt_mt5-clean_data_is-noise.csv")
    """
    map = {0: 4, 1: 3, 2: 0, 3: 2, 4: 1, 5: 6, 6: 5}
    mapping_label(
        "/data/ephemeral/home/gj/cluster-label_bt_mt5-clean_data_is-noise.csv", map
    )
