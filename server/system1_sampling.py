#! /usr/bin/python3

# Allow the three system generators to run in parallel.  Each system's
# response is an HTML string.
#
# To simplyfy Docker configuration, this code supports the following
# environment variables for locating model and vocabulary files:
#
# CWC_STORY_MODEL_SYSTEM_1
# CWC_STORY_VOCAB_SYSTEM_1
#
# There are two different System 1 implementations:  Beam and Sampling.
# This is the sampling implementation. 

import os
import sys
sys.path.append("../pytorch_src")

import torch
from utils import load_pickle, load_model, init_nlp_model
from interactive_generate import to_ints, generate, preprocess

import system_template

SPECIAL_CHARACTERS = {"<EOL>", "<eos>", "#", "<EOT>", "</s>", "<P>"}

class System1_Generator(system_template.Story_Generator):
    SAMPLING_MODEL_FILE = "ROC_title_e1000_h1500_edr0.2_hdr0.1_lr10.pt"
    SAMPLING_VOCAB_FILE = "ROC_title_e1000_h1500_edr0.2_hdr0.1_lr10.pkl"

    def __init__(self, system_id):
        super().__init__(system_id)

        self.story_model = os.environ.get("CWC_STORY_MODEL_" + system_id.upper(),
                                          self.model_folder + "/" + self.SAMPLING_MODEL_FILE)
        self.story_vocab = os.environ.get("CWC_STORY_VOCAB_" + system_id.upper(),
                                          self.model_folder + "/" + self.SAMPLING_VOCAB_FILE)
        torch.manual_seed(self.torch_seed)
        
        # Load models and vocab dictionaries, init stopping symbols for generation
        self.st_model = load_model(self.story_model, self.use_cuda)
        self.st_dict = load_pickle(self.story_vocab)
        self.st_vocab_size = len(self.st_dict)
        self.st_eot_id = self.st_dict.word2idx[self.title_end]
        self.st_eos_id = self.st_dict.word2idx[self.story_end]
        self.st_sep_id = self.st_dict.word2idx[self.story_sep]
        # self.special_chars = [self.kw_end, self.story_end, self.kw_sep, self.story_sep, self.title_end]
        self.special_chars = SPECIAL_CHARACTERS
        self.nlp = init_nlp_model()
        
    def generate_response(self, topic, kw_temp=None, story_temp=None, dedup=None, max_len=None, use_gold_titles=None, oov_handling=None):
        story_temp = 0.22
        print("%s: Processing %s (story_temp=%s, dedup=%s, max_len=%s, sampling)" % (self.system_id, topic, "NONE" if story_temp is None else str(story_temp), "NONE" if dedup is None else str(dedup), "NONE" if max_len is None else str(max_len)))

        #if story_temp is None:
        #    story_temp = 0.3 # Default value for this model
        #elif story_temp == 0.0:
        #    story_temp = 0.001 # Avoid division by 0

        if dedup is None:
            dedup = False

        if max_len is None or max_len == 0:
            max_len = 250

        topic = preprocess(topic, self.special_chars, self.nlp)
        topic = self.apply_oov(topic, self.st_dict.word2idx, oov_handling)
        story_prefix = to_ints(topic, self.st_dict)
        story_prefix.append(self.st_eot_id)

        story_phrases = generate(self.st_model, self.st_dict, story_prefix,
                                 self.st_eos_id, self.st_sep_id,
                                 max_len, dedup, self.use_cuda, story_temp)
        story = ' '.join(story_phrases)

        return  self.format_response(story)

