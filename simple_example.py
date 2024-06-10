from DPF import ShardsDatasetConfig, DatasetReader
from DPF.filters.images.info_filter import ImageInfoFilter
from DPF.filters.images.hash_filters import PHashFilter


if __name__ == "__main__":
    config = ShardsDatasetConfig.from_path_and_columns(
        'examples/example_dataset',
        image_name_col='image_name',
        text_col="caption"
    )

    reader = DatasetReader()
    processor = reader.read_from_config(config)

    print("Dataset summary:", processor.print_summary())

    datafilter = ImageInfoFilter(workers=16)
    print('Applying ImageInfoFilter')
    processor.apply_data_filter(datafilter)

    datafilter = PHashFilter(sim_hash_size=8, workers=16)
    print('Applying PHashFilter')
    processor.apply_data_filter(datafilter)

    print('Result dataset metadata')
    print(processor.df)