import torch
import math
from torch.utils.data import TensorDataset, DataLoader


def update_surr_model(
    model,
    mll,
    learning_rte,
    train_x,
    train_y,
    n_epochs
):
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_x.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model


def update_surrogate_models(
    models_list,
    mlls_list,
    learning_rte,
    train_x,
    train_y,
    n_epochs,
):
    updated_models = []
    for ix, model in enumerate(models_list):
        updated_model = update_surr_model(
            model,
            mll=mlls_list[ix],
            learning_rte=learning_rte,
            train_x=train_x,
            train_y=train_y[:,ix],
            n_epochs=n_epochs,
        )
        updated_models.append(updated_model)
    
    return updated_models

