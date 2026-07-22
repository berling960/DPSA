#!/usr/bin/env python3
"""
DPSA: DINOv3-Guided Path Sampling Attack.

Attack Generation:
    python main.py --attack dpsa --model dinov3 --num_attacks 100

Evaluation:
    python main.py --attack dpsa --model dinov3 --num_attacks 100 -e
"""
import argparse
import csv
import os
import torch
import tqdm
import transferattack
from transferattack.utils import *


def get_parser():
    parser = argparse.ArgumentParser(description='DPSA: DINOv3-Guided Path Sampling Attack')

    # --- Mode ---
    parser.add_argument('-e', '--eval', action='store_true',
                        help='evaluation mode: ASR on black-box models')
    parser.add_argument('--attack', default='dpsa', type=str,
                        choices=list(transferattack.attack_zoo.keys()),
                        help='attack algorithm')
    parser.add_argument('--model', default='dinov3', type=str,
                        help='surrogate model (dinov3, resnet50, vit_small, ...)')
    parser.add_argument('--dataset', default='ImageNet',
                        choices=['ImageNet', 'CUB', 'CUB1000', 'AIR', 'STCAR'],
                        help='dataset')
    parser.add_argument('--num_attacks', default=-1, type=int,
                        help='number of images (-1 = all)')
    parser.add_argument('--start_index', default=0, type=int,
                        help='dataset start index for attack')
    parser.add_argument('--input_dir', default='./data', help='input data directory')
    parser.add_argument('--output_dir', default='./results', help='output directory')
    parser.add_argument('--pretrained_models_root', default='./pretrained_models',
                        help='pretrained model/checkpoint root')
    parser.add_argument('--GPU_ID', default='0', help='CUDA device index')
    parser.add_argument('--batchsize', default=1, type=int, help='batch size')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--epoch', type=int, default=15, help='attack iterations')
    parser.add_argument('--eval_output_csv', default=None, help='path for eval CSV results')

    # --- OPS baseline parameters ---
    parser.add_argument('--num_sample_neighbor', type=int, default=None,
                        help='OPS neighbor perturbation samples')
    parser.add_argument('--num_sample_operator', type=int, default=None,
                        help='OPS operator samples')

    # --- DPSA: Core parameters ---
    parser.add_argument('--num_neighbor', type=int, default=15,
                        help='number of operator paths initialized from a neighborhood point')
    parser.add_argument('--num_op_samples', type=int, default=30,
                        help='operator paths per iteration')
    parser.add_argument('--pool_chain_length', type=int, default=3,
                        help='operators per chain')
    parser.add_argument('--operator_pool_variant', default='full',
                        choices=['full', 'unified', 'ops', 'basic', 'no_special'],
                        help='DPSA operator pool variant for ablation')
    parser.add_argument('--gds_diversity_weight', type=float, default=0.25,
                        help='GDS diversity component weight')
    parser.add_argument('--gds_loss_weight', type=float, default=0.3,
                        help='path quality weight used in GDS fusion')
    parser.add_argument('--local_probe_samples', type=int, default=1,
                        help='forward-only local probes per operator path endpoint')
    parser.add_argument('--local_probe_radius', type=float, default=0.05,
                        help='local probe radius as a multiple of epsilon')
    parser.add_argument('--local_probe_weight', type=float, default=0.25,
                        help='weight for local probe mean loss in GDS quality')
    parser.add_argument('--local_probe_std_weight', type=float, default=0.00,
                        help='penalty for local probe loss std in GDS quality')

    return parser.parse_args()


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    args.input_dir = os.path.join(args.input_dir, args.dataset)
    model_display = get_model_display_name(args.model)
    output_dir = os.path.join(args.output_dir, args.dataset, model_display, args.attack)
    os.makedirs(output_dir, exist_ok=True)

    dataset = AdvDataset(dataset=args.dataset, input_dir=args.input_dir,
                         output_dir=output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    # ============================================================
    # Attack Mode
    # ============================================================
    if not args.eval:
        if args.attack in ('dpsa', 'lpsa'):
            model = None
        else:
            if ',' in args.model:
                args.model = args.model.split(',')
            model = load_single_model(
                args.model, dataset_name=args.dataset,
                pretrained_models_root=args.pretrained_models_root
            )
            model = model.eval().cuda()

        if args.attack in ('dpsa', 'lpsa'):
            attacker_kwargs = {
                'num_neighbor': args.num_neighbor,
                'num_op_samples': args.num_op_samples,
                'pool_chain_length': args.pool_chain_length,
                'operator_pool_variant': args.operator_pool_variant,
                'gds_diversity_weight': args.gds_diversity_weight,
                'gds_loss_weight': args.gds_loss_weight,
                'local_probe_samples': args.local_probe_samples,
                'local_probe_radius': args.local_probe_radius,
                'local_probe_weight': args.local_probe_weight,
                'local_probe_std_weight': args.local_probe_std_weight,
                'dataset_name': args.dataset,
                'pretrained_models_root': args.pretrained_models_root,
                'epoch': args.epoch,
            }
        else:
            attacker_kwargs = {'epoch': args.epoch}
            if args.num_sample_neighbor is not None:
                attacker_kwargs['num_sample_neighbor'] = args.num_sample_neighbor
            if args.num_sample_operator is not None:
                attacker_kwargs['num_sample_operator'] = args.num_sample_operator

        attacker = transferattack.load_attack(
            args.attack, args.model, model, args.targeted, **attacker_kwargs)

        limit = args.num_attacks if args.num_attacks > 0 else len(dataloader)
        start_index = max(int(args.start_index), 0)
        attacked = 0
        pbar = tqdm.tqdm(desc=f'[Attack: {args.attack}]', total=limit)
        for batch_idx, (images, labels, filenames) in enumerate(dataloader):
            if batch_idx < start_index:
                continue
            if attacked >= limit:
                break
            perturbations = attacker(images, labels)
            save_images(output_dir, images + perturbations.cpu(), filenames)
            attacked += 1
            pbar.update(1)
        pbar.close()

    # ============================================================
    # Evaluation Mode
    # ============================================================
    else:
        eval_cnn = ['resnet50', 'densenet121', 'mobilenet_v2']
        eval_mlp = ['mixer_b16_224', 'resmlp_12_224']
        eval_vit = ['vit_small_patch16_224', 'swin_tiny_patch4_window7_224', 'deit3_small_patch16_224']
        eval_ssm = ['vmamba_small']
        eval_selfsup = ['dinov3-vits16', 'dinov2-vits14', 'mae-base']
        all_eval_models = eval_cnn + eval_mlp + eval_vit + eval_ssm + eval_selfsup

        existing_f2l = {}
        for fname in dataset.f2l:
            stem = os.path.splitext(fname)[0]
            if os.path.exists(os.path.join(output_dir, f"{stem}.png")):
                existing_f2l[fname] = dataset.f2l[fname]
        dataset.f2l = existing_f2l

        results = []
        for model_name, model in load_pretrained_model(
            all_eval_models, dataset_name=args.dataset,
            pretrained_models_root=args.pretrained_models_root
        ):
            model = model.eval().cuda()
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0
            for images, labels, _ in tqdm.tqdm(
                torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4),
                desc=f'[Eval {model_name}]'
            ):
                if args.targeted:
                    labels = labels[1]
                pred = model(images.cuda()).argmax(dim=1).detach().cpu()
                correct += (pred.numpy() == labels.numpy()).sum()
                total += labels.shape[0]
            asr = (1 - correct / total) * 100 if not args.targeted else (correct / total) * 100
            results.append((model_name, asr))
            print(f'{model_name}: {asr:.1f}')

        res_line = ' | '.join(f'{a:.1f}' for _, a in results)
        print(f'\nResults: | {res_line} |')

        if args.eval_output_csv:
            with open(args.eval_output_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['eval_model', 'asr'])
                for name, asr in results:
                    w.writerow([name, f'{asr:.4f}'])


if __name__ == '__main__':
    main()
