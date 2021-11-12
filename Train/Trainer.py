import torch
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
from .Evaluator import RankCollector
from torchviz import make_dot

class Trainer(object):

    def __init__(self, model=None, train=None, validation=None, train_times=1000, alpha=0.5, use_gpu=True,
                 opt_method="sgd", save_steps=None, checkpoint_dir=None, weight_decay=0, momentum=0, inner_norm = False):
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.model = model
        self.train = train
        self.validation = validation
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.patient_count = 3
        self.finished = False
        self.inner_norm = inner_norm

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        #self.model.startingBatch()
        
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        
        loss.backward()
        
        self.optimizer.step()
        return loss

    def run(self, init_epoch=0):
        patient = 0
        if self.use_gpu:
            self.model.cuda()
        
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        print("Finish initializing...")

        collector = None
        # Ranks and totals may have been computed before
        # If they exist, just collect and load them
        '''
        if os.path.exists(self.checkpoint_dir + ".ranks") and os.path.exists(self.checkpoint_dir + ".totals"):
            # Report metric
            
            with open(self.checkpoint_dir + ".ranks", 'rb') as f:
                all_ranks = np.load(f)
            with open(self.checkpoint_dir + ".totals", 'rb') as f:
                all_totals = np.load(f)
            collector = RankCollector()
            collector.load(self.model.model.ranks.tolist(), self.model.model.totals.tolist())
        '''

        #To check: Ranks
        if self.model.model.ranks!=None and self.model.model.totals!=None:
            collector = RankCollector()
            collector.load(self.model.model.ranks, self.model.model.totals)
            
        for epoch in range(init_epoch, self.train_times+1):
            if self.save_steps and self.checkpoint_dir and epoch > 0 and epoch % self.save_steps == 0:
                if self.validation is not None:
                    start = time.perf_counter()
                    new_collector = self.validation.evaluate(self.model.model)
                    end = time.perf_counter()
                    print("Validation metric: ", new_collector.get_metric(metric_str = "mrh").get(), "; Time:", end-start)

                # If the new collector did not significantly improve the previous one or random, stop!
                if collector is not None and ((new_collector.get_metric().is_improved(collector.get_metric()) and
                    new_collector.is_significant(collector.all_ranks)) or
                        (new_collector.get_metric().is_improved(new_collector.get_expected()) and
                            new_collector.is_significant_expected())):

                    
                    print('Previous metric value:', collector.get_metric().get(), " was not improved and is significant")
                    self.finished = True
                    break
                else:
                    
                    #If we are not finished, save model as .valid and save totals and ranks
                    #self.model.model.save_checkpoint(os.path.join(self.checkpoint_dir + ".valid"), epoch=epoch)
                    self.model.model.epoch = epoch
                    torch.save(self.model.model, os.path.join(self.checkpoint_dir + ".valid"))
                    '''
                    with open(os.path.join(self.checkpoint_dir + ".ranks"), 'wb') as f:
                        np.save(f, np.array(new_collector.all_ranks))
                    with open(os.path.join(self.checkpoint_dir + ".totals"), 'wb') as f:
                        np.save(f, np.array(new_collector.all_totals))
                    '''
                    self.model.model.ranks = new_collector.all_ranks
                    self.model.model.totals = new_collector.all_totals
                    collector = new_collector
                    print("Epoch %d has finished, saving..." % epoch)
                    #print (self.model.model.embeddings["entity"]["e"].emb.weight.data[0])
                    break

            if epoch < self.train_times:
                res = 0.0
                start = time.perf_counter()
                start_neg = time.perf_counter()
                time_neg = 0
                
                if not self.inner_norm:
                    self.model.model.normalize()
                
                for data in self.train:
                    
                    end_neg = time.perf_counter()
                    time_neg+=end_neg-start_neg
                    loss = self.train_one_step(data)
                    res += loss.item()
                    start_neg = time.perf_counter()
                    #make_dot(loss, params=dict(list(self.model.model.named_parameters()))).render(self.model.model.model_name, format="png")
                    #exit()

                #self.model.model.save_checkpoint(os.path.join(self.checkpoint_dir + ".ckpt"), epoch=epoch)
                self.model.model.epoch = epoch
                torch.save(self.model.model, os.path.join(self.checkpoint_dir + ".ckpt"))
                end = time.perf_counter()
                print("Epoch:",epoch,"; Loss:",res,"; Time:", end-start,"; Time neg.:",time_neg)

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
