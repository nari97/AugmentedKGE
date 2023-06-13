import math

import torch
import torch.optim as optim
import time
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, r2_score
from scipy.stats import ks_2samp, pearsonr

# TODO Save/load models with pickle:
# https://stackabuse.com/scikit-learn-save-and-restore-models/


class Calibrator(object):

    def __init__(self, model=None, managers=None, pos_target=None, neg_targets=None):
        self.model = model
        # These are the triple managers to use. If there are multiple, they use different corruption strategies, so we
        #   need to check all of them.
        if managers is None:
            managers = []
        self.managers = managers
        self.pos_target = pos_target
        # These are the fixed targets that we will use for each manager.
        if neg_targets is None:
            neg_targets = []
        self.neg_targets = neg_targets

    # positive_target_correction is the function used to correct positive targets (if any).
    # negative_target_correction is the function used to correct negative targets (if any).
    def train(self, positive_target_correction=None, negative_target_correction=None, use_weight=False):
        raise NotImplementedError

    def predict(self, scores):
        raise NotImplementedError

    def get_totals(self, triples):
        print('Computing totals...')
        total_positives, total_negatives = 0, 0
        start = time.perf_counter()
        for triple in triples:
            h, r, t = triple.h, triple.r, triple.t
            total_positives += 1

            # This will contain an entry with the corrupted heads/tails of each manager.
            corrupted_heads, corrupted_tails = {}, {}
            for i in range(len(self.managers)):
                corrupted_heads[i] = self.managers[i].get_corrupted(h, r, t, type='head')
                corrupted_tails[i] = self.managers[i].get_corrupted(h, r, t, type='tail')

            # The first manager has the highest priority and the last manager the lowest.
            for i in range(len(self.managers)):
                for j in range(i + 1, len(self.managers)):
                    corrupted_heads[j] -= corrupted_heads[i]
                    corrupted_tails[j] -= corrupted_tails[i]

            # Get totals and targets.
            heads, tails = [], []
            for i in range(len(self.managers)):
                heads += list(corrupted_heads[i])
                tails += list(corrupted_tails[i])
            total_negatives += len(heads) + len(tails)
        end = time.perf_counter()
        print('Total positives:', total_positives, '; Total negatives:', total_negatives, "; Time:", end - start)

        return total_positives, total_negatives

    def get_batch_and_targets(self, h, r, t):
        # This will contain an entry with the corrupted heads/tails of each manager.
        corrupted_heads, corrupted_tails = {}, {}
        for i in range(len(self.managers)):
            corrupted_heads[i] = self.managers[i].get_corrupted(h, r, t, type='head')
            corrupted_tails[i] = self.managers[i].get_corrupted(h, r, t, type='tail')

        # The first manager has the highest priority and the last manager the lowest.
        for i in range(len(self.managers)):
            for j in range(i + 1, len(self.managers)):
                corrupted_heads[j] -= corrupted_heads[i]
                corrupted_tails[j] -= corrupted_tails[i]

        # Get totals and targets.
        heads, tails = [], []
        target_heads, target_tails = None, None
        for i in range(len(self.managers)):
            heads += list(corrupted_heads[i])
            tails += list(corrupted_tails[i])

            these_target_heads = self.neg_targets[i].repeat(len(corrupted_heads[i]))
            these_target_tails = self.neg_targets[i].repeat(len(corrupted_tails[i]))

            if target_heads is None:
                target_heads = these_target_heads
                target_tails = these_target_tails
            else:
                target_heads = torch.cat((target_heads, these_target_heads))
                target_tails = torch.cat((target_tails, these_target_tails))

        batch_h = torch.cat((torch.tensor([h]), torch.tensor(heads), torch.tensor([h]).repeat(len(tails))))
        batch_r = torch.cat((torch.tensor([r]), torch.tensor([r]).repeat(len(heads)),
                             torch.tensor([r]).repeat(len(tails))))
        batch_t = torch.cat((torch.tensor([t]), torch.tensor([t]).repeat(len(heads)), torch.tensor(tails)))

        return {'batch_h': batch_h, 'batch_t': batch_t, 'batch_r': batch_r}, target_heads, target_tails

    @staticmethod
    def get_tensor(value, grad):
        return torch.tensor([value], dtype=torch.float64, requires_grad=grad)

    @staticmethod
    def get_weights(use_weight, bs, total_positives, total_negatives):
        if use_weight:
            return torch.cat((torch.tensor([1/total_positives]), torch.tensor([1/total_negatives]).repeat(bs-1)))
        else:
            return torch.tensor([1]).repeat(bs)

    def test(self, manager, only_score=False):
        start = time.perf_counter()

        # Get test triples.
        triples = manager.tripleList

        all_expected, all_scores, all_ranks, positive_probs = None, None, [], []
        # For each positive triple.
        for triple in triples:
            h, r, t = triple.h, triple.r, triple.t

            heads, tails = list(manager.get_corrupted(h, r, t, type='head')), \
                list(manager.get_corrupted(h, r, t, type='tail'))

            # For certain strategies, there are no negatives, skip!
            if len(heads) == 0 and len(tails) == 0:
                continue

            # Sometimes, these tensors are not long (why?); change them to long explicitly.
            batch_h = torch.cat((torch.tensor([h]), torch.tensor(heads), torch.tensor([h]).repeat(len(tails)))).to(
                torch.long)
            batch_r = torch.cat((torch.tensor([r]), torch.tensor([r]).repeat(len(heads)), torch.tensor([r]).repeat(
                len(tails)))).to(torch.long)
            batch_t = torch.cat((torch.tensor([t]), torch.tensor([t]).repeat(len(heads)), torch.tensor(tails))).to(
                torch.long)

            # Get scores.
            with torch.no_grad():
                scores_from_model = self.model({'batch_h': batch_h, 'batch_t': batch_t, 'batch_r': batch_r})

                # Predict the probability.
                scores = self.predict(scores_from_model)

                # Compute fractional ranks.
                positive_triple_score = scores_from_model[0].item()
                corrupted_heads, corrupted_tails = scores_from_model[1:len(heads)+1], scores_from_model[len(heads)+1:]
                rankh_less, rankt_less = torch.sum(positive_triple_score < corrupted_heads).item(), torch.sum(
                    positive_triple_score < corrupted_tails).item()
                rankh_eq, rankt_eq = 1 + torch.sum(positive_triple_score == corrupted_heads).item(), 1 + torch.sum(
                    positive_triple_score == corrupted_tails).item()

                # If it is NaN, rank last!
                if math.isnan(positive_triple_score):
                    rankh_less, rankt_less = len(corrupted_heads), len(corrupted_tails)
                    rankh_eq, rankt_eq = 1, 1

                def frac_rank(less, eq):
                    return (2 * less + eq + 1) / 2

                # The same probability score results into two different ranks.
                all_ranks += [frac_rank(rankh_less, rankh_eq), frac_rank(rankt_less, rankt_eq)]
                positive_probs += [scores[0].item(), scores[0].item()]

            # Positive first, then all negatives.
            expected = torch.cat((torch.tensor([1]), torch.tensor([0]).repeat(len(heads)),
                                  torch.tensor([0]).repeat(len(tails))))
            if all_expected is None:
                all_expected = expected
            else:
                all_expected = torch.cat((all_expected, expected))

            if all_scores is None:
                all_scores = scores
            else:
                all_scores = torch.cat((all_scores, scores))

        end = time.perf_counter()
        print('Test time:', end - start)

        # Extra tests.
        print('Computing extra stuff...')
        start = time.perf_counter()

        def get_weights(expected):
            total_pos, total_neg = expected[expected == 1].shape[0], expected[expected == 0].shape[0]

            weights = torch.zeros_like(expected, dtype=torch.float64)
            weights[expected == 1] = 1.0 / total_pos
            weights[expected == 0] = 1.0 / total_neg

            return weights

        # Get weights.
        all_expected_weights = get_weights(all_expected)
        # Get R2 score.
        r2_weighted = r2_score(all_expected, all_scores, sample_weight=all_expected_weights)

        if only_score:
            # Using R2 score to determine which model is better.
            return r2_weighted

        # Correlation between positive probabilities and ranks.
        correlation = pearsonr(all_ranks, positive_probs)

        roc_auc = roc_auc_score(all_expected, all_scores, sample_weight=all_expected_weights)
        # Brier scores
        bs_all_weighted = brier_score_loss(all_expected, all_scores, sample_weight=all_expected_weights)

        bs_pos, bs_neg = brier_score_loss(all_expected[all_expected == 1], all_scores[all_expected == 1]), \
            brier_score_loss(all_expected[all_expected == 0], all_scores[all_expected == 0])

        precision, recall, pr_thresholds = precision_recall_curve(all_expected, all_scores,
                                                                  sample_weight=all_expected_weights)
        this_auc = auc(recall, precision)

        # Kolmogorov-Smirnov test.
        ks_scores = None
        if len(all_scores[all_expected == 1]) != 0 and len(all_scores[all_expected == 0]) != 0:
            ks_scores = ks_2samp(all_scores[all_expected == 1], all_scores[all_expected == 0])

        cutoff = .5
        # Expected is one and the scores are greater/less than cutoff.
        all_positives, all_negatives = all_scores[all_expected == 1], all_scores[all_expected == 0]
        true_positives, false_negatives = all_positives[all_positives >= cutoff], \
            all_positives[all_positives < cutoff]
        # Expected is zero and the scores are less/greater than cutoff.
        false_positives, true_negatives = all_negatives[all_negatives >= cutoff], \
            all_negatives[all_negatives < cutoff]

        deemed_positives, deemed_negatives = torch.cat((true_positives, false_positives)), \
            torch.cat((true_negatives, false_negatives))

        # Kolmogorov-Smirnov test.
        ks_cutoff = None
        if len(deemed_positives) != 0 and len(deemed_negatives) != 0:
            ks_cutoff = ks_2samp(deemed_positives, deemed_negatives)

        def get_five_number_summary(data):
            data = data.detach().numpy()
            if data.size > 0:
                data_quartiles = np.percentile(data, [25, 50, 75])
                data_min, data_max = data.min(), data.max()
                print('Min:', data_min, '; Q1:', data_quartiles[0], '; Mean:', data_quartiles[1],
                      '; Q3:', data_quartiles[2], '; Max:', data_max)
            else:
                print('Empty data! No five-number summary.')

        # Five number summary.
        print('Five number summary positives')
        get_five_number_summary(deemed_positives)
        print('Five number summary negatives')
        get_five_number_summary(deemed_negatives)
        print('TP/FN/TN/FP')
        print(true_positives.shape[0], false_negatives.shape[0], true_negatives.shape[0], false_positives.shape[0])

        mr, mr_std = np.mean(all_ranks), np.std(all_ranks)
        mp, mp_std = np.mean(positive_probs), np.std(positive_probs)

        end = time.perf_counter()
        print('Time:', end - start)

        print('AUC score:', roc_auc)
        print('AUC:', this_auc)
        print('KS (using scores):', ks_scores)
        print('KS (using cutoff):', ks_cutoff)
        print('Brier score:', bs_all_weighted)
        print('Brier score positive and negative:', bs_pos, bs_neg)
        print('R2 score:', r2_weighted)
        print('Pearson correlation:', correlation)
        print('Mean rank:', mr, '+-', mr_std)
        print('Mean positive probability:', mp, '+-', mp_std)
        print('All ranks:', all_ranks)
        print('All positive probabilities:', positive_probs)

        return 'tp,fn,tn,fp,roc_auc,auc,ks_scores,ks_cutoff,brier,brier_pos,brier_neg,r2,pearson,mr,mr_std,mp,mp_std', \
            str(true_positives.shape[0])+','+str(false_negatives.shape[0])+','+str(true_negatives.shape[0])+',' +\
            str(false_positives.shape[0])+','+str(roc_auc)+','+str(this_auc)+','+str(ks_scores)+','+str(ks_cutoff)+',' +\
            str(bs_all_weighted)+','+str(bs_pos)+','+str(bs_neg)+','+str(r2_weighted)+','+str(correlation)+',' +\
            str(mr)+','+str(mr_std)+','+str(mp)+','+str(mp_std)


class PlattCalibrator(Calibrator):

    def __init__(self, model=None, managers=None, pos_target=None, neg_targets=None,
                 init_a_values=[1], init_b_values=[0]):
        super(PlattCalibrator, self).__init__(model, managers, pos_target, neg_targets)

        # These are the numbers we wish to learn with default values. We will train several at the same time.
        self.all_as, self.all_bs, self.all_inits = [], [], []
        for init_a_value in init_a_values:
            for init_b_value in init_b_values:
                self.all_as.append(Calibrator.get_tensor(init_a_value, True))
                self.all_bs.append(Calibrator.get_tensor(init_b_value, True))
                self.all_inits.append(str(init_a_value)+', '+str(init_b_value))
        # These will be the actual final values selected.
        self.a, self.b = None, None

    def predict(self, scores):
        return torch.sigmoid(self.forward(scores))

    # Do not use sigmoid!
    def forward(self, scores):
        return self.a * scores - self.b

    def train(self, positive_target_correction=None, negative_target_correction=None, use_weight=False):
        # Declare the optimizers using all initialization values. We will train several models at the same time.
        optimizers = []
        for idx in range(len(self.all_as)):
            optimizers.append(optim.Adam([self.all_as[idx], self.all_bs[idx]]))

        # Select a manager as the one that will provide the positive triples.
        triples = self.managers[0].tripleList

        # Let's compute the totals.
        total_positives, total_negatives = super().get_totals(triples)

        # This is the target correction of the positives.
        eps_plus = 0
        if positive_target_correction is not None:
            eps_plus = positive_target_correction(total_positives)

        # This is the target correction of the negatives.
        eps_minus = 0
        if negative_target_correction is not None:
            eps_minus = negative_target_correction(total_negatives)

        current, total_time = 0, 0
        for triple in triples:
            start = time.perf_counter()
            h, r, t = triple.h, triple.r, triple.t
            current += 1

            batch, target_heads, target_tails = super().get_batch_and_targets(h, r, t)
            weights = Calibrator.get_weights(use_weight, 1 + len(target_heads) + len(target_tails),
                                             total_positives, total_negatives)

            # Get scores.
            with torch.no_grad():
                scores_from_model = self.model(batch)

            # Learn the stuff! For each optimizer
            for idx, optimizer in enumerate(optimizers):
                self.a, self.b = self.all_as[idx], self.all_bs[idx]
                optimizer.zero_grad()

                # Compute scores using a and b.
                scores = self.forward(scores_from_model)

                # Adjust negative targets.
                target_heads, target_tails = target_heads + eps_minus, target_tails + eps_minus

                # Apply loss function and soft constraints.
                if use_weight:
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights)
                else:
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(scores, torch.cat((torch.tensor([self.pos_target - eps_plus]), target_heads, target_tails)))

                # Backward and optimize (the magic happens here!).
                loss.backward()
                optimizer.step()

            end = time.perf_counter()

            total_time += end - start
            if current % 50 == 0:
                print("Epoch:", current, " out of:", len(triples), "; Loss:", loss.item(), "; Time:", end - start,
                      "; As:", [a.item() for a in self.all_as], "; Bs: ", [a.item() for a in self.all_bs])

        # Select the best one among all calibration models using mean positive probabilities.
        best_mean, best_a, best_b, best_init = 0, None, None, None
        for idx in range(len(self.all_as)):
            self.a, self.b = self.all_as[idx], self.all_bs[idx]

            batch_h, batch_r, batch_t = [], [], []
            for triple in triples:
                batch_h.append(triple.h)
                batch_r.append(triple.r)
                batch_t.append(triple.t)

            with torch.no_grad():
                scores = self.model({'batch_h': torch.tensor(batch_h), 'batch_t': torch.tensor(batch_t),
                                     'batch_r': torch.tensor(batch_r)})
                calib_scores = self.predict(scores)

            this_mean = np.mean(calib_scores.detach().numpy())

            if this_mean > best_mean:
                best_a, best_b, best_mean, best_init = self.a, self.b, this_mean, self.all_inits[idx]

        print('Best init and positive mean:', best_init, best_mean)
        self.a, self.b = best_a, best_b

        print("Total training time:", total_time)


class IsotonicCalibrator(Calibrator):

    def __init__(self, model=None, managers=None, pos_target=None, neg_targets=None, init_a_value=1, init_b_value=0):
        super(IsotonicCalibrator, self).__init__(model, managers, pos_target, neg_targets)

        # This is the regressor we wish to learn.
        self.regressor = None

    def predict(self, scores):
        # Clamp the scores
        return torch.tensor(self.regressor.predict(torch.clamp(
            scores, min=self.regressor.X_min_, max=self.regressor.X_max_)))

    def train(self, positive_target_correction=None, negative_target_correction=None, use_weight=False):
        # Select a manager as the one that will provide the positive triples.
        triples = self.managers[0].tripleList

        # Let's compute the totals.
        total_positives, total_negatives = super().get_totals(triples)

        # This is the target correction of the positives.
        eps_plus = 0
        if positive_target_correction is not None:
            eps_plus = positive_target_correction(total_positives)

        # This is the target correction of the negatives.
        eps_minus = 0
        if negative_target_correction is not None:
            eps_minus = negative_target_correction(total_negatives)

        current, total_time = 0, 0
        all_x, all_y, all_weights = None, None, None
        for triple in triples:
            start = time.perf_counter()
            h, r, t = triple.h, triple.r, triple.t
            current += 1

            batch, target_heads, target_tails = super().get_batch_and_targets(h, r, t)
            weights = Calibrator.get_weights(use_weight, 1 + len(target_heads) + len(target_tails),
                                             total_positives, total_negatives)

            # Get scores.
            with torch.no_grad():
                scores = self.model(batch)

            cc = (torch.tensor([self.pos_target - eps_plus]), target_heads + eps_minus, target_tails + eps_minus)
            if all_x is None:
                all_x = scores
                all_y = torch.cat(cc)
                all_weights = weights
            else:
                all_x = torch.cat((scores, all_x))
                all_y = torch.cat(cc + (all_y,))
                all_weights = torch.cat((weights, all_weights))

            end = time.perf_counter()

            total_time += end - start
            if current % 50 == 0:
                print("Epoch:", current, " out of:", len(triples), "; Time:", end - start)

        print("Total batch time:", total_time)

        start = time.perf_counter()
        print('Fitting isotonic regressor...')
        self.regressor = IsotonicRegression().fit(all_x, all_y, sample_weight=all_weights)
        end = time.perf_counter()
        print("Time:", end - start)



