def log_config(model):
    print('transformer layers: {}'.format(model.enc_layers))
    print('transformer heads: {}'.format(model.transformer.nhead))
    print('hidden dim: {}'.format(model.decoder.embedding_dim))
    print('num classes: {}'.format(model.decoder.num_embeddings))
    print('backbone: {}'.format(model.backbone_name))
    print('dropout: {}'.format(model.pos_encoder.dropout.p))
    print(f'{count_parameters(model):,} trainable parameters')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
