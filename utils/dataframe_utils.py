import pandas as pd
import os


def format_dataframe(file_name, save=False):
    """
    ID, text, target 포맷으로 변경하는 함수
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_dir, f"./data/{file_name}.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"'{file_path}' 경로에 파일이 존재하지 않습니다.")

    df = pd.read_csv(file_path)

    # 노이즈를 복원한 경우, 복원된 텍스트 열의 이름은 clean_text로 해주세요
    if "clean_text" in df.columns:
        df["text"] = df["clean_text"]
        df.drop(columns=["clean_text"])
    if "is_noise" in df.columns:
        df = df[df["is_noise"] == 1.0]
        df.drop(columns=["is_noise"])

    df = df[["ID", "text", "target"]]

    if save:
        formatted_file_path = os.path.join(
            parent_dir, f"./data/formatted_{file_name}.csv"
        )
        df.to_csv(formatted_file_path, index=False, quoting=0)

    return df


def concat_df(file_name_list, save=False):
    """
    데이터 프레임의 포맷을 맞춘 뒤 concat하는 함수
    """
    formatted_df_list = []
    for file_name in file_name_list:
        try:
            df = format_dataframe(file_name)
            formatted_df_list.append(df)
        except:
            continue
    concatenated_df = pd.concat(formatted_df_list)

    if save:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        concat_file_path = os.path.join(parent_dir, f"./data/concat_data.csv")
        concatenated_df.to_csv(concat_file_path, index=False, quoting=0)

    return concatenated_df


def split_clean_and_noise_data(file_path):
    """
    is_noise가 1.0인 데이터만 필터링하는 함수
    """
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
