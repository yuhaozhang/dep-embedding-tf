from collections import Counter
import numpy as np
import os, sys

class DataOptions:
    def __init__(self):
        self.DATA_DIR = './data/'
        self.CONLL_FILE = 'test.conll'
        self.DEP_FILE = 'word_dep.txt'
        self.WORD_VOCAB_FILE = 'word.vocab'
        self.CONTEXT_VOCAB_FILE = 'context.vocab'

        self.FREQ_CUTOFF = 10

        # parameters for dependency
        self.SEP_SYMBOL = '@@'
        self.REV_SEP_SYMBOL = '@I@'
        self.ROOT = (0, '*root*', -1, 'rroot')
        self.CONLL_ARRAY_LENGTH = 10
        self.TARGET_POS = -4
        self.DEP_NAME_POS = -3

        self.compile()

    def compile(self):
        self.CONLL_PATH = os.path.join(self.DATA_DIR, self.CONLL_FILE)
        self.DEP_PATH = os.path.join(self.DATA_DIR, self.DEP_FILE)
        self.WORD_VOCAB_PATH = os.path.join(self.DATA_DIR, self.WORD_VOCAB_FILE)
        self.CONTEXT_VOCAB_PATH = os.path.join(self.DATA_DIR, self.CONTEXT_VOCAB_FILE)

class Data:
    def __init__(self, options):
        self.options = options
        self.word_dict = {}
        self.ctx_dict = {}
        self.word2idx = {}
        self.ctx2idx = {}
        self.idx2word = {}
        self.idx2ctx = {}

        self.word_vocab_size = 0
        self.ctx_vocab_size = 0
        self.total_examples = 0

        self.infile = None
        self.epoch = 0

        self.isprepared = False

    def build_dict_from_data(self, conll_file, cutoff):
        d = Counter()
        word_dict = {}
        with open(conll_file, 'r') as infile:
            for line in infile:
                array = line.split()
                if len(array) < 2: continue # empty line
                d[array[1].lower()] += 1
        for w,c in d.iteritems():
            if c>=cutoff:
                word_dict[w] = c
        return word_dict

    def yield_conll(self, conll_file, opt):
        tokens = [opt.ROOT]
        token_count = 0
        with open(conll_file, 'r') as infile:
            for line in infile:
                array = line.split()
                if len(array) < opt.CONLL_ARRAY_LENGTH: # encounter invalid line
                    if len(tokens) > 1: yield tokens
                    tokens = [opt.ROOT] # start new line
                    token_count = 0
                else:
                    token_count += 1
                    token_pos = int(array[0])
                    # Insert placeholder token if a token is missing at a position.
                    # This could happen when the dependency graph does not include punctuation or special symbols.
                    while token_pos > token_count:
                        tokens.append((token_count,'',-1,''))
                        token_count += 1
                    tokens.append((token_pos, array[1].lower(), 
                        int(array[opt.TARGET_POS]), array[opt.DEP_NAME_POS].lower()))
        if len(tokens) > 1:
            yield tokens

    def extract_deps(self, conll_file, dep_file, word_dict, cutoff):
        opt = self.options
        context_dict = Counter()
        with open(dep_file, 'w') as outfile:
            for i,sent in enumerate(self.yield_conll(conll_file, opt)):
                for token in sent[1:]:
                    if token[2] == -1: continue # skip placeholder token
                    t_token = sent[token[2]]
                    w = token[1]
                    if w not in word_dict: continue
                    rel = token[3]
                    if rel == 'adpmod': continue
                    # collapse preposition relations
                    if rel == 'adpobj' and t_token[0]!=0:
                        tt_token = sent[t_token[2]]
                        rel = '%s:%s' % (t_token[3],t_token[1])
                        t_w = tt_token[1]
                    else:
                        t_w = t_token[1]
                    if t_w not in word_dict: continue
                    forward_context = opt.SEP_SYMBOL.join((rel, w))
                    backward_context = opt.REV_SEP_SYMBOL.join((rel, t_w))
                    context_dict[forward_context] += 1
                    context_dict[backward_context] += 1
                    outfile.write(t_w + '\t' + forward_context + '\n')
                    outfile.write(w + '\t' + backward_context + '\n')
                    self.total_examples += 2
        pruned_context_dict = {}
        for w,c in context_dict.iteritems():
            if c >= cutoff:
                pruned_context_dict[w] = c
        return pruned_context_dict

    def write_dict(self, dict_list, dict_file_list):
        assert(len(dict_list) == len(dict_file_list))
        for d, df in zip(dict_list, dict_file_list):
            with open(df, 'w') as outfile:
                for w,c in d.iteritems():
                    outfile.write(w+'\t'+str(c)+'\n')
        return

    def get_word_mappings(self):
        for i,w in enumerate(self.word_dict.keys()):
            self.word2idx[w] = i
            self.idx2word[i] = w
        for i,w in enumerate(self.ctx_dict.keys()):
            self.ctx2idx[w] = i
            self.idx2ctx[i] = w
        return

    def prepare_data(self):
        opt = self.options
        # build vocabularies
        self.word_dict = self.build_dict_from_data(opt.CONLL_PATH, opt.FREQ_CUTOFF)
        self.word_vocab_size = len(self.word_dict)
        print 'Word dictionary loaded, with vocabulary size: %d.' % self.word_vocab_size

        self.ctx_dict = self.extract_deps(opt.CONLL_PATH, opt.DEP_PATH, self.word_dict, opt.FREQ_CUTOFF)
        self.ctx_vocab_size = len(self.ctx_dict)
        print 'Word-context pairs extraction done, with context vocabulary size: %d.' % self.ctx_vocab_size

        # write dicts to files
        self.write_dict([self.word_dict, self.ctx_dict], [opt.WORD_VOCAB_PATH, opt.CONTEXT_VOCAB_PATH])
        # create indexes for words and contexts
        self.get_word_mappings()
        print "Data is now ready."
        self.isprepared = True
        return

    def create_input_stream(self):
        self.infile = open(self.options.DEP_PATH, 'r')

    def get_batch(self, batch_size):
        if self.infile == None:
            self.create_input_stream()
        batch_words = []
        batch_contexts = []
        left = batch_size
        while left > 0:
            line = self.infile.readline()
            if len(line) == 0: # EOF, start a new epoch
                self.create_input_stream()
                line = self.infile.readline()
                self.epoch += 1
            t = line.strip().split()
            if len(t) < 2: continue
            w,c = t[0], t[1]
            if w not in self.word2idx or c not in self.ctx2idx: continue
            batch_words.append(self.word2idx[w])
            batch_contexts.append(self.ctx2idx[c])
            left -= 1
        # reshape the context vector so that it is compatible with the model
        batch_contexts = np.asarray(batch_contexts).reshape([batch_size, 1])
        return (batch_words, batch_contexts)

if __name__=='__main__':
    opt = DataOptions()
    data = Data(opt)
    data.prepare_data()
    print data.word_vocab_size
    print data.ctx_vocab_size

    batch_w, batch_c = data.get_batch(10)
    print batch_w
    print batch_c
    

