"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import csv
import time
import json
import datetime
import math
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):
    def _append_metrics_csv(self, epoch, train_stats, test_stats):
        if not dist_utils.is_main_process():
            return

        row = OrderedDict()
        row["epoch"] = epoch
        for key in sorted(train_stats.keys()):
            row[f"train_{key}"] = float(train_stats[key])
        for key in sorted(test_stats.keys()):
            value = test_stats[key]
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    row[f"{key}_{idx}"] = float(item)
            else:
                row[key] = float(value)

        if not hasattr(self, "_metrics_csv_fields"):
            self._metrics_csv_fields = list(row.keys())

        csv_path = self.output_dir / "metrics.csv"
        file_exists = csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._metrics_csv_fields,
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _save_plots(self, epoch_label: int):
        if not dist_utils.is_main_process():
            return

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve().parents[2] / "tools" / "visualization" / "plot_training_log.py"
        log_path = self.output_dir / "log.txt"
        summary_dir = self.output_dir / "summary"
        
        if not log_path.exists():
            print(f"[Plot] Log file not found: {log_path}, skipping plot generation")
            return
        
        cmd = [
            sys.executable,
            str(script_path),
            "--log",
            str(log_path),
            "--summary-dir",
            str(summary_dir),
            "--out-dir",
            str(plot_dir),
            "--smooth",
            "10",
        ]
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"[Plot] Successfully generated plots in {plot_dir}")
            else:
                print(f"[Plot] Plot generation returned code {result.returncode}")
                if result.stderr:
                    print(f"[Plot] Error: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print(f"[Plot] Plot generation timed out")
        except Exception as exc:  # noqa: BLE001
            print(f"[Plot] Failed to generate plots: {exc}")

    def _safe_load_stage_checkpoint(self, path, fallback_path=None, retries=3, sleep_sec=2.0):
        """Safely load stage checkpoint in distributed training."""
        last_error = None
        for attempt in range(1, retries + 1):
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()
            try:
                self.load_resume_state(str(path))
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                return True
            except Exception as exc:  # noqa: BLE001 - log and retry
                last_error = exc
                print(f"[Checkpoint] Failed to load {path} (attempt {attempt}/{retries}): {exc}")
                time.sleep(sleep_sec)

        if fallback_path is not None:
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()
            try:
                self.load_resume_state(str(fallback_path))
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                print(f"[Checkpoint] Loaded fallback {fallback_path}")
                if dist_utils.is_main_process():
                    dist_utils.save_on_master(self.state_dict(), path)
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                return True
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[Checkpoint] Failed to load fallback {fallback_path}: {exc}")

        print(f"[Checkpoint] Continue without loading {path}. Last error: {last_error}")
        return False

    def fit(self, ):
        self.train()
        args = self.cfg
        plot_freq = args.yaml_cfg.get('plot_freq', 50)
        try:
            plot_freq = int(plot_freq)
        except (TypeError, ValueError):
            plot_freq = 50
        if plot_freq <= 0:
            plot_freq = None

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self._safe_load_stage_checkpoint(
                    self.output_dir / 'best_stg1.pth',
                    fallback_path=self.output_dir / 'last.pth',
                )
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every checkpoint_freq epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # Update best statistics
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                current_ap = test_stats[k][0]
                
                if k in best_stat:
                    if current_ap > best_stat[k]:
                        best_stat['epoch'] = epoch
                        best_stat[k] = current_ap
                        print(f'[Best Model] New best {k}: {current_ap:.6f} at epoch {epoch} (previous: {best_stat_print.get(k, 0):.6f})')
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = current_ap

                # Save best model if improved
                if current_ap > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = current_ap
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                            print(f'[Checkpoint] Saved best_stg2.pth with AP {current_ap:.6f}')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
                            print(f'[Checkpoint] Saved best_stg1.pth with AP {current_ap:.6f}')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

            # Handle stage transition (only once when crossing stop_epoch threshold)
            stop_epoch = self.train_dataloader.collate_fn.stop_epoch
            if stop_epoch > 0 and epoch == stop_epoch:
                best_stg1_path = self.output_dir / 'best_stg1.pth'
                if best_stg1_path.exists():
                    print(f'[Stage Transition] Loading best_stg1.pth at epoch {epoch}')
                    self._safe_load_stage_checkpoint(
                        best_stg1_path,
                        fallback_path=self.output_dir / 'last.pth',
                    )
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
                else:
                    print(f'[Stage Transition] best_stg1.pth not found at epoch {epoch}, continuing without reset')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # Generate plots and CSV every plot_freq epochs
                if plot_freq and (epoch + 1) % plot_freq == 0:
                    print(f'[Plot] Generating plots at epoch {epoch + 1}...')
                    self._save_plots(epoch + 1)
                
                # Save CSV every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self._append_metrics_csv(epoch, train_stats, test_stats)

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state