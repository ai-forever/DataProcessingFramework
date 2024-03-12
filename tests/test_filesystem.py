from DPF.connectors import S3Connector


def test_s3_join():
    fs = S3Connector('test', 'test', 'test')

    assert fs.join('s3://example-bucket/path/to/dataset', 'shards/') == \
           's3://example-bucket/path/to/dataset/shards'
    assert fs.join('s3://example-bucket/path/to/dataset/', 'shards/') == \
           's3://example-bucket/path/to/dataset/shards'

    assert fs.join('s3://example-bucket/path/to/dataset/shards', '1.tar') == \
           's3://example-bucket/path/to/dataset/shards/1.tar'
    assert fs.join('s3://example-bucket/path/to/dataset/', 'shards', '1.tar') == \
           's3://example-bucket/path/to/dataset/shards/1.tar'
