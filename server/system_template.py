#! /usr/bin/python3

import html
import os

import nltk

from interactive_generate import replace_oovs

class Story_Generator:
    def __init__(self, system_id, apply_disc=False):
        self.system_id =system_id

        use_cuda = os.environ.get("CWC_USE_CUDA", "FALSE")
        if use_cuda.upper() in ["TRUE", "YES", "Y"]:
            self.use_cuda = True
        else:
            self.use_cuda = False
            
        self.torch_seed = int(os.environ.get("CWC_TORCH_SEED", "1111"))
        self.model_folder = os.environ.get("CWC_MODEL_FOLDER", "../models")
        self.data_folder = os.environ.get("CWC_DATA_FOLDER", "../data/")

        print_cond_data = os.environ.get("CWC_PRINT_COND_DATA", "FALSE")
        if print_cond_data.upper() in ["TRUE", "YES", "Y"]:
            self.print_cond_data = True
        else:
            self.print_cond_data = False

        self.kw_temp = float(os.environ.get("CWC_KW_TEMP", "0.5"))
        self.kw_end = os.environ.get("CWC_KW_END", "<EOL>")
        self.kw_sep = os.environ.get("CWC_KW_SEP", "#")

        story_temp = os.environ.get("CWC_STORY_TEMP", "NONE")
        if story_temp.upper() == "NONE":
            self.story_temp = None
        else:
            self.story_temp = float(story_temp)

        self.story_end = os.environ.get("CWC_STORY_END", "<eos>")
        self.story_sep = os.environ.get("CWC_STORY_SEP", "</s>")
        self.story_unk = os.environ.get("CWC_STORY_UNK", "<unk>")
        self.beam_size = int(os.environ.get("CWC_BEAM_SIZE", "5"))
        self.title_end = os.environ.get("CWC_TITLE_END", "<EOT>")

        cwc_apply_disc = os.environ.get("CWC_APPLY_DISC", None)
        if cwc_apply_disc is None:
            self.apply_disc = apply_disc
        else:
            if cwc_apply_disc.upper() in ["TRUE", "YES", "Y"]:
                self.apply_disc = True
            else:
                self.apply_disc = False

        self.nltk_resource_path = os.environ.get("CWC_NLTK_RESOURCE_PATH", "../nltk-data")
        nltk.data.path = [ self.nltk_resource_path ]

        # self.oov_handling = None # Do not substitute for out-of-vocabulary words.
        self.oov_handling = "wordnet" # Use wordnet to substitute for out-of-vocabulary words.

    def apply_oov(self, topic, word2idx, oov_handling):
        if oov_handling is None:
            # print("%s: using self.oov_handling" % self.system_id)
            oov_handling = self.oov_handling

        if oov_handling is None:
            print("%s: Skipping OOV." % self.system_id)
            return topic

        mod_topic = replace_oovs(topic, word2idx, oov_handling, self.special_chars)
        if topic != mod_topic:
            print("%s: topic modified to: %s" % (self.system_id, mod_topic))

        return mod_topic


    def format_response(self, response):
        lines = [ ]
        for line in response.split(self.story_sep):
            if len(line) > 0:
                lines.append(html.escape(line.replace(self.kw_sep, "->").replace(self.kw_end, "").replace(self.story_end, "")))
        return "<br>".join(lines)

    def format_response2(self, response):
        return response.replace(self.kw_sep, "->").replace(self.kw_end, "").replace(self.story_end, "")


    
