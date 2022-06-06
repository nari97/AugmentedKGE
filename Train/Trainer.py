import torch
import time
from .Evaluator import RankCollector


class Trainer(object):

    def __init__(self, loss=None, train=None, validation=None, train_times=1000, save_steps=None,
                 checkpoint_dir=None, load_valid=None, save_valid=None, save_checkpoint=None,
                 optimizer=None):
        self.train_times = train_times

        self.optimizer = optimizer

        self.loss = loss
        self.train = train
        self.validation = validation
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.patient_count = 3
        self.finished = False

        self.load_valid = load_valid
        self.save_valid = save_valid
        self.save_checkpoint = save_checkpoint

    def train_one_step(self, data):
        # Apply normalization.
        self.loss.model.start_batch()

        self.optimizer.zero_grad()

        def to_var(x):
            return torch.LongTensor(x)

        # Inner norm happens in the forward of the Model.
        loss = self.loss({
            'batch_h': to_var(data['batch_h']),
            'batch_t': to_var(data['batch_t']),
            'batch_r': to_var(data['batch_r']),
            'batch_y': to_var(data['batch_y'])
        })

        loss.backward()

        self.optimizer.step()

        self.loss.model.end_batch()

        return loss

    def run(self, init_epoch=0):
        # Get ranks and totals from the valid model.
        collector = None

        # Get previous model after validation (if any):
        prev_valid = self.load_valid()
        if prev_valid is not None:
            collector = RankCollector()
            collector.load(prev_valid.ranks, prev_valid.totals)
            
        for epoch in range(init_epoch+1, self.train_times+1):
            with torch.no_grad():
                if self.save_steps and self.checkpoint_dir and epoch > 0 and epoch % self.save_steps == 0:
                    if self.validation is not None:
                        start = time.perf_counter()
                        new_collector = self.validation.evaluate(self.loss.model)
                        end = time.perf_counter()
                        print("Validation metric: ", new_collector.get_metric().get(), "; Time:", end-start)

                    # If the new collector did not significantly improve the previous one or random, stop!
                    if new_collector.stop_train(collector):
                        print('Previous metric value:', collector.get_metric().get(), " was not improved and is significant")
                        self.finished = True
                        break
                    else:
                        # If we are not finished, save model as .valid and save totals and ranks
                        self.loss.model.epoch = epoch
                        self.loss.model.ranks = new_collector.all_ranks
                        self.loss.model.totals = new_collector.all_totals

                        # Save validation model.
                        self.save_valid()

                        # Update the collector and keep going
                        collector = new_collector
                        print("Epoch %d has finished, saving..." % epoch)

            if epoch < self.train_times:
                res = 0.0
                start = time.perf_counter()
                start_neg = time.perf_counter()
                time_neg = 0

                for data in self.train:
                    end_neg = time.perf_counter()
                    time_neg += end_neg - start_neg
                    res += self.train_one_step(data).item()
                    start_neg = time.perf_counter()

                self.loss.model.epoch = epoch
                # Save checkpoint model.
                self.save_checkpoint()
                end = time.perf_counter()
                print("Epoch:",epoch,"; Loss:",res,"; Time:", end-start,"; Time neg.:",time_neg)
