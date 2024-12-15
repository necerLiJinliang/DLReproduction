from model import CTransR
from dataset import TransRDataset
import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm


def train(args, model, dataset, device):
    model.train()
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=TransRDataset.collate_fn,
        num_workers=20,
        shuffle=True,
    )
    epochs = args.epochs
    optimizer = torch.optim.AdamW(
        [
            {"params": model.entity_embeddings.parameters(), "lr": 0.001},
            {"params": model.relation_embeddings.parameters(), "lr": 0.001},
            {"params": model.relation_proj.parameters(), "lr": 0.001},
        ],
        lr=args.lr,
    )
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    if args.load_path:
        optimizer.load_state_dict(
            torch.load(args.load_path, map_location=device)["optimizer_state_dict"],
        )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.000001,
        total_iters=epochs,
    )
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc="Training"):
            inputs = {
                "h": batch["h"].to(device),
                "r": batch["r"].to(device),
                "t": batch["t"].to(device),
                "neg_samples": batch["neg_samples"].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 5 == 0:
            # torch.save(model.state_dict(), args.save_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.save_path,
            )
        print(f"Epoch:{epoch+1},loss:{total_loss / len(data_loader)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TransR model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4800,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--entity_dim",
        type=int,
        default=50,
        help="Dimension of the embeddings",
    )
    parser.add_argument(
        "--relation_dim",
        type=int,
        default=50,
        help="Dimension of the embeddings",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1,
        help="Margin for the loss function",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.01,
        help="Margin for the loss function",
    )
    parser.add_argument(
        "--norm",
        type=int,
        default=1,
        help="Margin for the loss function",
    )
    parser.add_argument(
        "--filtered",
        type=bool,
        default=False,
        help="Whether filter the corrupted data from all triples",
    )
    parser.add_argument(
        "--advance",
        type=bool,
        default=False,
        help="Whether filter the corrupted data from all triples",
    )
    parser.add_argument(
        "--bern",
        type=bool,
        default=False,
        help="Whether filter the corrupted data from all triples",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="8_TransR/dataset/FB15k",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="8_TransR/model_save/CTransR-model-unif.pth",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        # default="8_TransR/model_save/model-unif_filter.pth",
        default=None,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--cluster_matrix_path",
        type=str,
        default="8_TransR/dataset/FB15k/label_matrix.pt",
        help="Path to cluster matrix labels",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="alpha",
    )
    parser.add_argument(
        "--num_cluster",
        type=int,
        default=8,
        help="Num of cluster",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_data_path = os.path.join(args.data_folder, "train.txt")
    entity2id_path = os.path.join(args.data_folder, "entity2id.json")
    label2id_path = os.path.join(args.data_folder, "label2id.json")
    all_triples_path = os.path.join(args.data_folder, "all_triples.pkl")
    train_dataset = TransRDataset(
        data_dir=train_data_path,
        entity2id_file=entity2id_path,
        relation2id_file=label2id_path,
        triples_file=all_triples_path,
        mode="train",
        filtered=args.filtered,
        bern=args.bern,
        num_neg_samples=20,
        advance=args.advance,
    )
    cluster_ids_matrix = torch.load(args.cluster_matrix_path).long().to(device)
    num_entities = len(train_dataset.entity2id)
    num_relations = len(train_dataset.relation2id)
    model: CTransR = CTransR(
        entity_dim=args.entity_dim,
        relation_dim=args.relation_dim,
        margin=args.margin,
        num_entities=num_entities,
        num_relations=num_relations,
        norm=args.norm,
        c=args.c,
        alpha=args.alpha,
        num_cluster=args.num_cluster,
        cluster_ids_matrix=cluster_ids_matrix,
    ).to(device)
    transE_weight = torch.load("8_TransR/model_save/TransE-model-unif.pth")[
        "model_state_dict"
    ]
    with torch.no_grad():
        model.entity_embeddings.weight.copy_(
            torch.nn.Parameter(
                transE_weight["entity_embeddings.weight"], requires_grad=True
            )
        )
        model.relation_embeddings.weight.copy_(
            torch.nn.Parameter(
                transE_weight["relation_embeddings.weight"], requires_grad=True
            )
        )
    model.entity_embeddings.requires_grad_(True)
    model.relation_embeddings.requires_grad_(True)
    model.init_cluster_relation_embeddings()
    try:
        if args.load_path:
            print(f"Load model from {args.load_path}")
            model.load_state_dict(
                torch.load(args.load_path, map_location=device)["model_state_dict"]
            )
        train(args=args, model=model, dataset=train_dataset, device=device)
    finally:
        print(1)


if __name__ == "__main__":
    main()
