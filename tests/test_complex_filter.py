from DPF import DatasetReader
from DPF.configs import ShardsDatasetConfig
from DPF.filters import ComplexDataFilter
from DPF.filters.images.hash_filters import PHashFilter
from DPF.filters.images.info_filter import ImageInfoFilter


def test_shards_complex_filter():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    phashfilter = PHashFilter(workers=1)
    infofilter = ImageInfoFilter(workers=1)

    datafilter = ComplexDataFilter([phashfilter, infofilter], workers=2)
    dataset.apply_data_filter(datafilter)

    assert not dataset.df['image_phash_8'].isna().any()
    assert dataset.df['is_correct'].all()
    assert dataset.df['error'].isna().all()
    assert not dataset.df['width'].isna().any()
    assert not dataset.df['height'].isna().any()
    assert not dataset.df['channels'].isna().any()
