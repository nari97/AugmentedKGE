class Amie():
    def __init__(self):
        self.triples = {}

    def load_checkpoint(self, file):
        with open(file, 'r') as f:
            for line in f.readlines():
                s, p, o, score = line.split(sep='\t')
                s, p, o, score = int(s), int(p), int(o), float(score)

                if p not in self.triples.keys():
                    self.triples[p] = {}

                if s not in self.triples[p].keys():
                    self.triples[p][s] = {}

                self.triples[p][s][o] = score

    def predict(self, batch):
        scores = []
        for i in range(len(batch['batch_h'])):
            s, p, o = batch['batch_h'][i].item(), batch['batch_r'][i].item(), batch['batch_t'][i].item()
            try:
                score = self.triples[p][s][o]
            except:
                score = .0
            scores.append(1.0-score)
        return scores