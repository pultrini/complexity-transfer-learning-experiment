from dataclasses import dataclass


@dataclass(frozen=True)
class LMDatasetInfo:
    """File-path metadata for a text dataset used in language model training.

    Unlike vision datasets, text datasets here are not auto-downloaded: the
    files must already exist under ``data_root`` at the given relative paths.
    """

    train_file: str
    val_file: str
    test_file: str

LM_DATASET_REGISTRY: dict[str, LMDatasetInfo] = {
    "wikitext2": LMDatasetInfo(
        train_file="wikitext-2/wiki.train.tokens",
        val_file="wikitext-2/wiki.valid.tokens",
        test_file="wikitext-2/wiki.test.tokens",
    ),
    "tiny_shakespeare": LMDatasetInfo(
        train_file="tiny-shakespare/train.csv",
        val_file="tiny-shakespare/validation.csv",
        test_file="tiny-shakespare/test.csv",
    ),
}
