def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize FederatedDataset once
    global fds
    if fds is None:
        df = pd.read_parquet("./Ids_dataset/train_test_network_preprocessed.parquet")
        usrdataset = Dataset.from_pandas(df)
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = usrdataset
        fds = partitioner

    dataset = fds.load_partition(partition_id).with_format("pandas")[:]
    # Divide data on each node: 80% train, 20% test
    dataset.dropna(inplace=True)
    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader