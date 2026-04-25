from SoccerNet.Downloader import SoccerNetDownloader


def download_features(local_directory, files, split):
    """
    Downloads specified feature files from SoccerNet.

    local_directory : path to save the data
    files           : list of filenames to download
    split           : list of splits to download
    """
    downloader = SoccerNetDownloader(
        LocalDirectory=local_directory
    )
    downloader.downloadGames(files=files, split=split)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SoccerNet features"
    )
    parser.add_argument("--data_path", type=str,
                        default="D:/soccernet-data")
    parser.add_argument("--features", type=str,
                        nargs="+",
                        default=["1_baidu_soccer_embeddings.npy",
                                 "2_baidu_soccer_embeddings.npy"])
    parser.add_argument("--split", type=str,
                        nargs="+",
                        default=["train", "valid", "test"])
    args = parser.parse_args()

    print(f"Downloading to: {args.data_path}")
    print(f"Files: {args.features}")
    print(f"Splits: {args.split}")

    download_features(
        local_directory=args.data_path,
        files=args.features,
        split=args.split
    )

    print("Download complete.")