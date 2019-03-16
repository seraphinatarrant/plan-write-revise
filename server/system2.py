#! /usr/bin/python3

# Allow the three system generators to run in parallel.  Each system's
# response is an HTML string.
#
# To simplyfy Docker configuration, this code supports the following
# environment variables for locating model and vocabulary files:
#
# CWC_KEYWORD_MODEL_SYSTEM_2
# CWC_KEYWORD_VOCAB_SYSTEM_2
# CWC_STORY_MODEL_SYSTEM_2
# CWC_STORY_VOCAB_SYSTEM_2
# CWC_SCORERS_CONFIG_SYSTEM_2

import os
import sys
sys.path.append("../pytorch_src")

import torch
from utils import batchify, get_batch, repackage_hidden, load_pickle, load_model, init_nlp_model
from decoder.decoders import BeamRerankDecoder, BeamSearchDecoder
from interactive_generate import to_ints, generate, read_scorers, read_gold_storylines, preprocess
from mosestokenizer import MosesDetokenizer

import system_template

SPECIAL_CHARACTERS = {"<EOL>", "<eos>", "#", "<EOT>", "</s>", "<P>"}

class System2_Generator(system_template.Story_Generator):
    def __init__(self, system_id, apply_disc=False):
        super().__init__(system_id, apply_disc=apply_disc)

        self.keyword_model = os.environ.get("CWC_KEYWORD_MODEL_" + system_id.upper(),
                                            self.model_folder + "/ROC_title_keyword_e500_h1000_edr0.4_hdr0.1_511_lr10.pt")
        self.keyword_vocab = os.environ.get("CWC_KEYWORD_VOCAB_" + system_id.upper(),
                                            self.model_folder + "/ROC_title_keyword_e500_h1000_edr0.4_hdr0.1_511_lr10.pkl")
        self.story_model = os.environ.get("CWC_STORY_MODEL_" + system_id.upper(),
                                          self.model_folder + "/ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10.pt")
        self.story_vocab = os.environ.get("CWC_STORY_VOCAB_" + system_id.upper(),
                                          self.model_folder + "/ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10.pkl")
        self.scorers_config = os.environ.get("CWC_SCORERS_CONFIG_" + system_id.upper(),
                                          self.model_folder + "/scorer_weights_abl.tsv")
        self.gold_titles = os.environ.get("CWC_GOLD_TITLES_" + system_id.upper(),
                                          self.data_folder + "/ROCStories_all_merge_tokenize.titlesepkey.all")
        
        torch.manual_seed(self.torch_seed)
        
        # Load models and vocab dictionaries, init stopping symbols for generation
        self.kw_model = load_model(self.keyword_model, self.use_cuda)
        self.st_model = load_model(self.story_model, self.use_cuda)
        self.kw_dict = load_pickle(self.keyword_vocab)
        self.st_dict = load_pickle(self.story_vocab)
        self.kw_vocab_size = len(self.kw_dict)
        self.st_vocab_size = len(self.st_dict)
        self.st_eos_id = self.st_dict.word2idx[self.story_end]
        self.st_unk_id = self.st_dict.word2idx[self.story_unk]
        #self.kw_eos_id = self.kw_dict.word2idx[self.story_end] this is clearly wrong but seems to not be used ever
        self.kw_eot_id = self.kw_dict.word2idx[self.title_end]
        self.kw_end_id = self.kw_dict.word2idx[self.kw_end]
        self.kw_sep_id = self.kw_dict.word2idx[self.kw_sep]
        self.st_sep_id = self.st_dict.word2idx[self.story_sep]
        # self.special_chars = [self.kw_end, self.story_end, self.kw_sep, self.story_sep, self.title_end]
        self.title2storyline = read_gold_storylines(self.gold_titles, self.title_end)
        self.special_chars = SPECIAL_CHARACTERS
        self.nlp = init_nlp_model()
        self.detokenizer = MosesDetokenizer('en')

        if self.apply_disc:
            print("%s: Using BeamRerankDecoder" % (self.system_id))
            scorers, coefs = read_scorers(self.scorers_config, self.use_cuda)
            self.decoder = BeamRerankDecoder(self.st_model,
                                             scorers,
                                             coefs,
                                             beam_size=self.beam_size,
                                             sep=self.st_sep_id,
                                             temperature=None,
                                             terms=[self.st_eos_id],
                                             forbidden=[self.st_unk_id, self.st_eos_id],
                                             use_cuda=self.use_cuda)
        else:
            print("%s: Using BeamSearchDecoder" % (self.system_id))
            self.decoder = BeamSearchDecoder(self.st_model, self.beam_size, self.st_eos_id,
                                             verbosity=False, dictionary=self.st_dict, sep=self.st_sep_id)

    def generate_response(self, topic, kw_temp=None, story_temp=None, dedup=None, max_len=None, use_gold_titles=None, oov_handling=None):
        print("%s: Processing %s (kw_temp=%s, story_temp=%s, dedup=%s, max_len=%s, use_gold_titles=%s)" % (self.system_id, topic, "NONE" if kw_temp is None else str(kw_temp), "NONE" if story_temp is None else str(story_temp), "NONE" if dedup is None else str(dedup), "NONE" if max_len is None else str(max_len), "NONE" if use_gold_titles is None else str(use_gold_titles)))

        if dedup is None:
            dedup = True

        if max_len is None or max_len == 0:
            max_len = 25

        if use_gold_titles is None:
            use_gold_titles = True

        topic = preprocess(topic, self.special_chars, self.nlp)
        topic = self.apply_oov(topic, self.kw_dict.word2idx, oov_handling)
        topic_str = ' '.join(topic)
        title = to_ints(topic, self.kw_dict) + [self.kw_eot_id]  # title ends with EOT

        if use_gold_titles and topic_str in self.title2storyline.keys():
            print("%s: Using gold title storyline:" % (self.system_id))
            # title storyline mappings are not guaranteed to be unique. Currently picks first, could code differently
            storyline = self.title2storyline[topic_str][0]
        else:
            print("%s: Generating storyline from model:" % (self.system_id))
            # Storyline ends with EOL
            all_tokens = generate(self.kw_model, self.kw_dict, title, self.kw_end_id,
                                  self.kw_sep_id, max_len, dedup, self.use_cuda, kw_temp)
            storyline = ' '.join(all_tokens)
            
        topic_and_storyline = topic_str + " " + self.title_end + " " + storyline
        print("%s: %s" % (self.system_id, topic_and_storyline))
        story_prefix = to_ints(topic_and_storyline.split(), self.st_dict)

        with torch.no_grad():
            tokens = self.decoder.decode(story_prefix, temperature=story_temp)

        cont_tokens = [self.st_dict.idx2word[tokens[j]] for j in range(len(story_prefix), len(tokens))]
        if self.print_cond_data:
            init_tokens = [self.st_dict.idx2word[tokens[i]] for i in range(len(story_prefix))]
            cont_tokens = init_tokens + cont_tokens
        story = ' '.join(cont_tokens)

        # detokenize
        story = self.detokenizer(story.split())
        storyline = self.detokenizer(storyline.split())

        formatted_storyline = self.format_response(storyline)
        formatted_story = self.format_response(story)
        
        return "<h2>Storyline</h2>%s<h2>Story</h2>%s" % (formatted_storyline, formatted_story)

    def generate_storyline(self, topic, kw_temp=None, dedup=None, max_len=None, use_gold_titles=None, oov_handling=None):
        if dedup is None:
            dedup = True

        if max_len is None or max_len == 0:
            max_len = 25

        if use_gold_titles is None:
            use_gold_titles = True

        topic = preprocess(topic, self.special_chars, self.nlp)
        topic = self.apply_oov(topic, self.kw_dict.word2idx, oov_handling)
        topic_str = ' '.join(topic)

        print(
            "%s: Generating a storyline for  \"%s\" (kw_temp=%s, dedup=%s, max_len=%s, use_gold_titles=%s)" % (
            self.system_id, topic, "NONE" if kw_temp is None else str(kw_temp),
            "NONE" if dedup is None else str(dedup), "NONE" if max_len is None else str(max_len),
            "NONE" if use_gold_titles is None else str(use_gold_titles)))

        title = to_ints(topic, self.kw_dict) + [self.kw_eot_id]  # title ends with EOT

        if use_gold_titles and topic_str in self.title2storyline.keys():
            print("%s: Using gold title storyline:" % (self.system_id))
            # title storyline mappings are not guaranteed to be unique. Currently picks first, could code differently
            storyline = self.title2storyline[topic_str][0]
        else:
            print("%s: Generating storyline from model:" % (self.system_id))
            # Storyline ends with EOL
            all_tokens = generate(self.kw_model, self.kw_dict, title, self.kw_end_id,
                                  self.kw_sep_id, max_len, dedup, self.use_cuda, kw_temp)
            storyline = ' '.join(all_tokens)

        #detokenize
        storyline = self.detokenizer(storyline.split())
        print("%s: storyline: %s" % (self.system_id, storyline))
        return self.format_response2(storyline)

    def collab_storyline(self, topic, current_storyline, kw_temp=None, dedup=None, max_len=None, oov_handling=None):
        if dedup is None:
            dedup = True

        if max_len is None or max_len == 0:
            max_len = 25

        topic = preprocess(topic, self.special_chars, self.nlp)
        topic = self.apply_oov(topic, self.kw_dict.word2idx, oov_handling)
        topic_str = " ".join(topic)

        current_storyline = [phrase.strip() for phrase in current_storyline]

        if len(current_storyline) > 0:
            current_storyline_with_sep = (" " + self.kw_sep + " ").join(current_storyline) + " " + self.kw_sep
            # Storyline formatting for models
            print("Before",current_storyline_with_sep)
            current_storyline_with_sep = preprocess(current_storyline_with_sep, self.special_chars, self.nlp)
            current_storyline_with_sep = self.apply_oov(current_storyline_with_sep, self.kw_dict.word2idx, oov_handling)
            print("After", current_storyline_with_sep)
        else:
            current_storyline_with_sep = ""

        #TODO fix this print statement with the awkward joins
        print("%s: Collaborating on %s %s %s %s (kw_temp=%s, dedup=%s)" % (
            self.system_id, topic, self.title_end, current_storyline_with_sep,
            self.kw_sep if len(current_storyline) > 0 else " ",
            "NONE" if kw_temp is None else str(kw_temp),
            "NONE" if dedup is dedup is None else str(dedup)))

        used_word_ints = set()
        current_tokens = to_ints(topic, self.kw_dict)
        current_tokens.append(self.kw_eot_id) # title ends with EOT
        print("Current storyline %s" % current_storyline)
        if current_storyline_with_sep:
            phrase_tokens = to_ints(current_storyline_with_sep, self.kw_dict)
            used_word_ints.update(set(phrase_tokens))
            current_tokens.extend(phrase_tokens)
        #print("max len %s" % max_len)
        #print(current_tokens)
        new_words = generate(self.kw_model, self.kw_dict, current_tokens,
                             self.kw_end_id, self.kw_sep_id,
                             max_len, dedup, self.use_cuda, kw_temp,
                             only_one=True, forbidden=used_word_ints)
        if len(new_words) == 0:
            # This shouldn't happen, right?
            print ("%s: Returning with nothing to add." % (self.system_id))
            return {
                "new_phrase": "",
                "end_flag": True
            }

        last_word = new_words.pop() # removes the kw sep TODO would be better to not remove here and just mask in UI
        valid_ends = {self.kw_end, self.kw_sep}
        if last_word not in valid_ends:
            print("%s is not a valid storyline end token. Valid tokens: [%s]" % (last_word, valid_ends))
        end_flag = last_word == self.kw_end
        new_phrase = " ".join(new_words)
        print ("%s: Returning %s (end_flag=%s last_word=%s)" % (self.system_id, new_phrase, end_flag, last_word))
        return {
            "new_phrase": new_phrase,
            "end_flag": end_flag
        }

    def generate_story(self, topic, storyline_phrases, story_temp=None, oov_handling=None):
        topic = preprocess(topic, self.special_chars, self.nlp)
        topic = self.apply_oov(topic, self.kw_dict.word2idx, oov_handling)
        topic_str = ' '.join(topic)
        topic_and_storyline = topic_str + " " + self.title_end
        # print("%s: len(storyline_phrases)=%d" % (self.system_id, len(storyline_phrases)))
        if len(storyline_phrases) > 0:
            storyline = (" " + self.kw_sep + " ").join(storyline_phrases) + " " + self.kw_end
            topic_and_storyline += " " + storyline + " </s> " #TODO this is to match the hardcoding of this in interactive mode since the UI there doesn't hid special symbols. Fix in future.


        print("%s: Generating a story for %s (story_temp=%s)" % (self.system_id, topic_and_storyline, "NONE" if story_temp is None else str(story_temp)))
        story_prefix = to_ints(topic_and_storyline.split(), self.st_dict)
        #print(story_prefix)

        with torch.no_grad():
            if self.system_id != "system_3":
                tokens = self.decoder.decode(story_prefix, temperature=story_temp)
            else:
                tokens = self.decoder.decode(story_prefix, temperature=story_temp, min_sentences=4)
        #print(tokens)

        cont_words = [self.st_dict.idx2word[tokens[j]] for j in range(len(story_prefix), len(tokens))]
        if self.print_cond_data:
            init_words = [self.st_dict.idx2word[tokens[i]] for i in range(len(story_prefix))]
            cont_words = init_words + cont_words

        #detokenize
        story = self.detokenizer(cont_words)

        print("%s: Generated story: %s" % (self.system_id, story))

        # formatted_storyline = self.format_response(storyline)
        # formatted_story = self.format_response(story)
        # return "<h2>Storyline</h2>%s<h2>Story</h2>%s" % (formatted_storyline, formatted_story)
        return  {
            "story": self.format_response(story)
        }

    def generate_interactive_story(self, topic, storyline_phrases, story_sentences,
                                   story_temp=None, max_len=None, oov_handling=None, only_one=False):
        topic_storyline_story = topic + " " + self.title_end
        return_story = ""
        #
        if len(storyline_phrases) > 0:
            storyline = (self.kw_sep).join(storyline_phrases) + " " + self.kw_end
            topic_storyline_story += (" " + storyline.strip())

        if len(story_sentences) > 0:
            return_story = " </s> ".join(story_sentences)  # this is for returning later.
            story = "</s> " + return_story # we ONLY want the leading in the non-returned version...because the story split messes everything up
            topic_storyline_story += (" " + story)

        print("Before Processing: ", topic_storyline_story)
        topic_storyline_story = preprocess(topic_storyline_story, self.special_chars, self.nlp)
        topic_storyline_story = self.apply_oov(topic_storyline_story, self.st_dict.word2idx, oov_handling)
        print("After Processing: ",topic_storyline_story)
        if len(story_sentences) < 5:
            topic_storyline_story.append("</s>")  # even if no sentences yet, appends to the end of the title just so that it doesn't show up in UI. SPECIFIC to how ROC training data is.

            print("%s: Generating a story for %s (story_temp=%s)" % (self.system_id, topic_storyline_story, "NONE" if story_temp is None else str(story_temp)))
            story_prefix = to_ints(topic_storyline_story, self.st_dict)

            with torch.no_grad():
                tokens = self.decoder.decode(story_prefix, temperature=story_temp, only_one=only_one)

            continuation = [self.st_dict.idx2word[tokens[j]] for j in range(len(story_prefix), len(tokens))]
            # defaults to false - I don't think it's necessary in our current UI
            if self.print_cond_data:
                init_words = [self.st_dict.idx2word[tokens[i]] for i in range(len(story_prefix))]
                continuation = init_words + continuation

            new_line = " ".join(continuation)
            #for line in story_sentences:
            #    new_line = line + ' </s> ' + new_line
            # return the story prefix and continuation if there was a story prefix, else just the continuation
            if return_story:
                return_story += " </s> " + new_line
            else:
                return_story = new_line
        else:
            return_story = story

        return_story = self.detokenizer(return_story.split())

        print ("%s: Returning %s)" % (self.system_id, return_story))
        return {
            "story": return_story,
        }
        
