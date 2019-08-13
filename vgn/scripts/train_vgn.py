from __future__ import print_function

import argparse
import os

import open3d
import torch
import torch.nn.functional as F
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.utils import tensorboard

from vgn.dataset import VGNDataset
from vgn.models import get_model


def loss_fn(score_pred, score):
    loss = F.binary_cross_entropy(score_pred, score)
    return loss


def _prepare_batch(batch, device):
    tsdf, idx, score = batch
    tsdf = tsdf.to(device)
    idx = idx.to(device)
    score = score.squeeze().to(device)
    return tsdf, idx, score


def _select_pred(out, idx):
    score_pred = torch.cat(
        [t[0, i, j, k].unsqueeze(0) for t, (i, j, k) in zip(out, idx)])
    return score_pred


def create_trainer(model, optimizer, loss_fn, device):
    model.to(device)

    def _update(_, batch):
        model.train()
        optimizer.zero_grad()
        tsdf, idx, score = _prepare_batch(batch, device)
        out = model(tsdf)
        score_pred = _select_pred(out, idx)
        loss = loss_fn(score_pred, score)
        loss.backward()
        optimizer.step()
        return loss

    return Engine(_update)


def create_evaluator(model, loss_fn, device):
    model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            tsdf, idx, score = _prepare_batch(batch, device)
            out = model(tsdf)
            score_pred = _select_pred(out, idx)
            return score_pred, score

    engine = Engine(_inference)

    return engine


def create_summary_writers(model, data_loader, device, log_dir):
    train_path = os.path.join(log_dir, 'train')
    val_path = os.path.join(log_dir, 'validation')

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    # Write the graph to Tensorboard
    trace, _, _ = _prepare_batch(next(iter(data_loader)), device)
    train_writer.add_graph(model, trace)

    return train_writer, val_writer


def train(args):
    device = torch.device('cuda')
    kwargs = {'pin_memory': True}

    # Create log directory for the current setup
    descr = 'model={},data={},batch_size={},lr={:.0e},seed={}'.format(
        args.model,
        args.data,
        args.batch_size,
        args.lr,
        args.seed,
    )
    log_dir = os.path.join(args.log_dir, descr)

    assert not os.path.exists(log_dir), 'log with this setup already exists'

    # Load and inspect data
    path = os.path.join('data/datasets', args.data, 'train')
    dataset = VGNDataset(path, args.rebuild_cache)

    train_size = int((1 - args.validation_split) * len(dataset))
    validation_size = int(args.validation_split * len(dataset))

    print('Size of training dataset: {}'.format(train_size))
    print('Size of validation dataset: {}'.format(validation_size))

    # Create train and validation data loaders
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    **kwargs)

    # Build the network
    model = get_model(args.model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create Ignite engines for training and validation
    trainer = create_trainer(model, optimizer, loss_fn, device)
    evaluator = create_evaluator(model, loss_fn, device)

    # Define metrics
    def thresholded_output_transform(output):
        score_pred, score = output
        score_pred = torch.round(score_pred)
        return score_pred, score

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    Loss(loss_fn).attach(evaluator, 'loss')
    Accuracy(thresholded_output_transform).attach(evaluator, 'acc')

    # Setup loggers and checkpoints
    ProgressBar(persist=True, ascii=True).attach(trainer, ['loss'])

    train_writer, val_writer = create_summary_writers(model, train_loader,
                                                      device, log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(_):
        evaluator.run(validation_loader)

        epoch = trainer.state.epoch
        train_loss = trainer.state.metrics['loss']
        val_loss = evaluator.state.metrics['loss']
        val_acc = evaluator.state.metrics['acc']

        train_writer.add_scalar('loss', train_loss, epoch)
        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('accuracy', val_acc, epoch)

    checkpoint_handler = ModelCheckpoint(
        log_dir,
        'best',
        score_function=lambda engine: -engine.state.metrics['loss'],
        n_saved=1,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        to_save={'model': model},
    )

    # Run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='model',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='dataset',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='path to log directory',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='input batch size for training (default: 32)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate (default: 1e-3)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)',
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='ratio of data used for validation (default: 0.2)',
    )
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), 'ERROR: cuda is not available'

    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()