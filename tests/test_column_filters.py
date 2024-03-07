from DPF import DatasetReader
from DPF.configs import FilesDatasetConfig, ShardsDatasetConfig
from DPF.filters.texts.lang_filter import LangFilter
from DPF.filters.texts.regex_filter import RegexFilter
from DPF.filters.texts.regexs import ENG_REGEXS, SPECIAL_REGEXS


def test_shards_langid_filter():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    columnfilter = LangFilter(workers=1)
    dataset.apply_column_filter(columnfilter)

    assert 'lang' in dataset.columns and 'lang_score' in dataset.columns
    assert not dataset.df['lang'].isna().any()
    assert not dataset.df['lang_score'].isna().any()


def test_shards_regex_filter():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    columnfilter = RegexFilter(ENG_REGEXS + SPECIAL_REGEXS, workers=1)
    dataset.apply_column_filter(columnfilter)

    assert 'clean_caption' in dataset.columns
    assert not dataset.df['clean_caption'].isna().any()


def test_files_info_filter():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_path_and_columns(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    columnfilter = LangFilter(workers=1)
    dataset.apply_column_filter(columnfilter)

    assert 'lang' in dataset.columns and 'lang_score' in dataset.columns
    assert not dataset.df['lang'].isna().any()
    assert not dataset.df['lang_score'].isna().any()


def test_files_regex_filter():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_path_and_columns(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    columnfilter = RegexFilter(ENG_REGEXS + SPECIAL_REGEXS, workers=1)
    dataset.apply_column_filter(columnfilter)

    assert 'clean_caption' in dataset.columns
    assert not dataset.df['clean_caption'].isna().any()
