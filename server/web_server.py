#! /usr/bin/env python3

# This is a minimal Flask-based Web server for the three-system
# story generator.

import torch.multiprocessing as mp

import argparse

from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from flask_cors import CORS

import nltk

import os
import time

import system1_beam as system1
import system2
import system3

app=Flask(__name__)
CORS(app) # Make the browser happier about cross-domain references.

# Tell the browser to discard cached static files after
# 300 seconds.  This will facilitate rapid development
# (for even more rapid development, set the value lower).
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300

AUTOMATIC_HTML_FILE="index.html"
INTERACTIVE_HTML_FILE="interactive.html"

NLTK_RESOURCE_PATH="../nltk_data"

SYSTEM_1_ID = "system_1"
SYSTEM_2_ID = "system_2"
SYSTEM_3_ID = "system_3"

# Future advanced interface options:
KW_TEMP = 0.5
STORY_TEMP = None

# The server debug mode state.
server_debug_mode = False


@app.route("/")
def read_root():
    """Supply the top-level HTML file."""
    return render_template(AUTOMATIC_HTML_FILE, server_debug_mode=server_debug_mode)


@app.route("/index.html")
def read_index():
    """Supply the top-level HTML file."""
    return render_template(AUTOMATIC_HTML_FILE, server_debug_mode=server_debug_mode)


@app.route("/interactive.html")
def read_interactive():
    """Supply the interactive HTML file."""
    return render_template(INTERACTIVE_HTML_FILE, server_debug_mode=server_debug_mode)


@app.route("/api/generate", methods=["GET", "POST"])
def generate():
    request_id = request.values.get("id", "")
    topic = request.values.get("topic", "")
    systems_default = " ".join([ SYSTEM_1_ID, SYSTEM_2_ID, SYSTEM_3_ID ])
    systems = request.values.get("systems", systems_default).split()

    kw_temp = request.values.get("kw_temp", KW_TEMP)
    if str(kw_temp).upper() == "NONE":
        kw_temp = None
    elif str(kw_temp) == "":
        kw_temp = KW_TEMP
    else:
        kw_temp = float(kw_temp)
        if kw_temp == 0.0: # Protect against divide-by-zero.
            kw_temp = 0.001

    story_temp = request.values.get("story_temp", STORY_TEMP)
    if str(story_temp).upper() == "NONE":
        story_temp = None
    elif str(story_temp) == "":
        story_temp = STORY_TEMP
    else:
        story_temp = float(story_temp)
        if story_temp == 0.0: # Protect against divide-by-zero.
            story_temp = 0.001

    dedup = request.values.get("dedup", "")
    if dedup == "":
        dedup = True
    elif dedup.upper() == "TRUE":
        dedup = True
    else:
        dedup = False

    max_len = request.values.get("max_len", "")
    if max_len.upper() == "NONE":
        max_len = None
    elif max_len == "":
        max_len = None
    else:
        max_len = int(max_len)

    use_gold_titles = request.values.get("use_gold_titles", "NONE")
    if use_gold_titles.upper() == "NONE":
        use_gold_titles = None
    elif use_gold_titles == "":
        use_gold_titles = None
    elif use_gold_titles.upper() == "TRUE":
        use_gold_titles = True
    elif use_gold_titles.upper() == "FALSE":
        use_gold_titles = False

    # print("Id=%s Topic=%s" % (request_id, topic))

    # We'd like to record the elapsed wall clock time that
    # it takes to process this request.
    start_time = time.perf_counter() # replaces time.clock()
    
    for system_id in systems:
        start_generation(system_id, topic, kw_temp, story_temp, dedup, max_len, use_gold_titles)

    response = {
        "n_story": request_id, # Nominally the number of stories generated.
        "storey_id": request_id # Nominally an ID for this story.
    }
    for system_id in systems:
        response[system_id] = get_response(system_id)

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)
    
    return jsonify(response)
    

@app.route("/api/generate_storyline", methods=["GET", "POST"])
def generate_storyline():
    request_id = request.values.get("id", "")
    topic = request.values.get("topic", "")
    systems_default = " ".join([ SYSTEM_1_ID, SYSTEM_2_ID, SYSTEM_3_ID ])
    systems = request.values.get("systems", systems_default).split()

    kw_temp = request.values.get("kw_temp", KW_TEMP)
    if str(kw_temp).upper() == "NONE":
        kw_temp = None
    elif str(kw_temp) == "":
        kw_temp = KW_TEMP
    else:
        kw_temp = float(kw_temp)
        if kw_temp == 0.0: # Protect against divide-by-zero.
            kw_temp = 0.001

    dedup = request.values.get("dedup", "")
    if dedup == "":
        dedup = True
    elif dedup.upper() == "TRUE":
        dedup = True
    else:
        dedup = False

    max_len = request.values.get("max_len", "")
    if max_len.upper() == "NONE":
        max_len = None
    elif max_len == "":
        max_len = None
    else:
        max_len = int(max_len)

    use_gold_titles = request.values.get("use_gold_titles", "NONE")
    if use_gold_titles.upper() == "NONE":
        use_gold_titles = None
    elif use_gold_titles == "":
        use_gold_titles = None
    elif use_gold_titles.upper() == "TRUE":
        use_gold_titles = True
    elif use_gold_titles.upper() == "FALSE":
        use_gold_titles = False

    # print("Id=%s Topic=%s" % (request_id, topic))

    # We'd like to record the elapsed wall clock time that
    # it takes to process this request.
    start_time = time.perf_counter() # replaces time.clock()
    
    
    for system_id in systems:
        start_storyline_generation(system_id, topic, kw_temp, dedup, max_len, use_gold_titles)

    response = {
        "n_story": request_id, # Nominally the number of stories generated.
        "storey_id": request_id # Nominally an ID for this story.
    }
    for system_id in systems:
        response[system_id] = get_response(system_id)

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)
    
    return jsonify(response)
    

@app.route("/api/collab_storyline", methods=["GET", "POST"])
def collab_storyline():
    request_id = request.values.get("id", "")
    system_id = request.values.get("system_id", SYSTEM_2_ID)
    topic = request.values.get("topic", "")
    storyline = request.values.get("storyline", "")
    if len(storyline) > 0:
        current_storyline = storyline.split("->")
    else:
        current_storyline = [ ]

    kw_temp = request.values.get("kw_temp", KW_TEMP)
    if str(kw_temp).upper() == "NONE":
        kw_temp = None
    elif str(kw_temp) == "":
        kw_temp = KW_TEMP
    else:
        kw_temp = float(kw_temp)
        if kw_temp == 0.0: # Protect against divide-by-zero.
            kw_temp = 0.001

    dedup = request.values.get("dedup", "")
    if dedup == "":
        dedup = True
    elif dedup.upper() == "TRUE":
        dedup = True
    else:
        dedup = False

    max_len = request.values.get("max_len", "")
    if max_len.upper() == "NONE":
        max_len = None
    elif max_len == "":
        max_len = None
    else:
        max_len = int(max_len)

    start_time = time.perf_counter() # replaces time.clock()
    start_collab_storyline(system_id, topic, current_storyline, kw_temp, dedup, max_len)

    response = get_response(system_id)
    response["request_id"] = request_id

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)
    
    return jsonify(response)


@app.route("/api/generate_interactive_story", methods=["GET", "POST"])
def generate_interactive_story():
    request_json = request.get_json(force=True)
    request_id = request_json.get('id')
    system_id = request_json.get('system_id')
    topic = request_json.get('topic')
    storyline = request_json.get('storyline')
    use_sentence = request_json.get('use_sentence')
    sentences = request_json.get('sentences')
    change_max_idx = request_json.get('change_max_idx')
    only_one = request_json.get('only_one')

    # request_id = request.values.get("id", "")
    # system_id = request.values.get("system_id", SYSTEM_2_ID)
    # topic = request.values.get("topic", "")
    # storyline = request.values.get("storyline", "")
    if len(storyline) > 0:
        storyline_phrases = storyline.split("->")
    else:
        storyline_phrases = []

    print(request_json)
    story_temp = request_json.get("story_temp", STORY_TEMP)
    if str(story_temp).upper() == "NONE":
        story_temp = None
    elif str(story_temp) == "":
        story_temp = STORY_TEMP
    else:
        story_temp = float(story_temp)
        if story_temp == 0.0: # Protect against divide-by-zero.
            story_temp = 0.001

    print(story_temp)

    start_time = time.perf_counter() # replaces time.clock()
    start_generate_interactive_story(system_id, topic, storyline_phrases, story_temp, use_sentence, sentences, change_max_idx, only_one)

    response = get_response(system_id)
    response["request_id"] = request_id

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)
    print(response)

    return jsonify(response)
    

@app.route("/api/generate_story", methods=["GET", "POST"])
def generate_story():
    request_id = request.values.get("id", "")
    system_id = request.values.get("system_id", SYSTEM_2_ID)
    topic = request.values.get("topic", "")
    storyline = request.values.get("storyline", "")
    if len(storyline) > 0:
        storyline_phrases = storyline.split("->")
    else:
        storyline_phrases = [ ]

    story_temp = request.values.get("story_temp", STORY_TEMP)
    if str(story_temp).upper() == "NONE":
        story_temp = None
    elif str(story_temp) == "":
        story_temp = STORY_TEMP
    else:
        story_temp = float(story_temp)
        if story_temp == 0.0: # Protect against divide-by-zero.
            story_temp = 0.001

    start_time = time.perf_counter() # replaces time.clock()
    start_generate_story(system_id, topic, storyline_phrases, story_temp)

    response = get_response(system_id)
    response["request_id"] = request_id

    # Record the end time, compute the elapsed seconds as a floating point
    # number, and format with two decimal points.
    end_time = time.perf_counter()
    response["elapsed"] = "{:.2f}".format(end_time - start_time)

    return jsonify(response)


# write auto mode data to txt
@app.route("/api/write_auto_txt", methods=["GET", "POST"])
def write_auto_txt():
    request_json = request.get_json(force=True)
    with open("auto_mode_logging.txt", "a") as out:
        out.write("New auto generation data:" + "\n")
        out.write("Story Topic:" + "\n")
        out.write(request_json.get('topic') + "\n")
        out.write("Storyline:" + "\n")
        out.write(request_json.get('storyline') + "\n")
        out.write("System1_story:" + "\n")
        out.write(request_json.get('system1_story') + "\n")
        out.write("System2_story:" + "\n")
        out.write(request_json.get('system2_story') + "\n")
        out.write("System3_story:" + "\n")
        out.write(request_json.get('system3_story') + "\n")

    return "success"


# write interactive mode data to txt
@app.route("/api/write_interactive_txt", methods=["GET", "POST"])
def write_interactive_txt():
    request_json = request.get_json(force=True)
    with open("interactive_mode_logging.txt", "a") as out:
        out.write("New interactive generation data:" + "\n")
        out.write("Story Topic:" + "\n")
        out.write(request_json.get('topic') + "\n")
        out.write("Storyline:" + "\n")
        out.write(request_json.get('storyline') + "\n")

        kw_temp = request_json.get('kw_temp')
        if kw_temp == "":
            kw_temp = "None"
        out.write("kw_temp: " + kw_temp + "\n")

        story_temp = request_json.get('story_temp')
        if story_temp == "":
            story_temp = "None"
        out.write("story_temp: " + story_temp + "\n")

        out.write("generate story:" + "\n")
        out.write(request_json.get('story') + "\n")

    return "success"



# Allow the three system generators to run in parallel.  Each system's
# response is an HTML string, which we send back to the parent using
# a Queue.
#
# This design creates a permanent process for each system, initializing
# the generator in that process.

request_queues = {}
result_queues = {}


def initialize_generator(story_generator_class, system_id):
    request_queue = mp.Queue()
    request_queues[system_id] = request_queue
    result_queue = mp.Queue()
    result_queues[system_id] = result_queue
    system_process = mp.Process(target=system_worker, args=(system_id, story_generator_class, request_queue, result_queue))
    system_process.start()


def initialize_generators():
    initialize_generator(system1.System1_Generator, SYSTEM_1_ID)
    initialize_generator(system2.System2_Generator, SYSTEM_2_ID)
    initialize_generator(system3.System3_Generator, SYSTEM_3_ID)


def start_generation(system_id, topic, kw_temp, story_temp, dedup, max_len, use_gold_titles):
    """Ask a system to start generating a storyline and story response."""
    worker_request = {
        "action": "generate",
        "topic": topic,
        "kw_temp": kw_temp,
        "story_temp": story_temp,
        "dedup": dedup,
        "max_len": max_len,
        "use_gold_titles": use_gold_titles
    }
    request_queues[system_id].put(worker_request)


def start_storyline_generation(system_id, topic, kw_temp, dedup, max_len, use_gold_titles):
    """Ask a system to start generating a storyline."""
    worker_request = {
        "action": "generate_storyline",
        "topic": topic,
        "kw_temp": kw_temp,
        "dedup": dedup,
        "max_len": max_len,
        "use_gold_titles": use_gold_titles
    }
    request_queues[system_id].put(worker_request)


def start_collab_storyline(system_id, topic, storyline, kw_temp, dedup, max_len):
    """Ask a system to collaboratively generate a storyline."""
    worker_request = {
        "action": "collab_storyline",
        "topic": topic,
        "storyline": storyline,
        "kw_temp": kw_temp,
        "dedup": dedup,
        "max_len": max_len
    }
    request_queues[system_id].put(worker_request)


def start_generate_interactive_story(system_id, topic, storyline, story_temp, use_sentence, sentences, change_max_idx, only_one):
    """Ask a system to start generating a story."""
    pass_sentences = []
    # append non-empty sentences to be used as a prefix for later story generation

    # so if the third sentence is deleted, change_max_idx will be 2, and i will be 0,1,2. Basically I think
    # it is the index beyond which we can generate new content.
    # so pass sentences will be variable length and contain everything to keep.
    # if a user modifies 0 and 3, 3 will be passed
    if use_sentence:
        for i in range(change_max_idx + 1):
            if len(sentences[i]) > 0:
                pass_sentences.append(sentences[i])

    worker_request = {
        "action": "generate_interactive_story",
        "topic": topic,
        "storyline": storyline,
        "story_sentences": pass_sentences,
        "story_temp": story_temp,
        "only_one": only_one
    }
    print(worker_request)
    request_queues[system_id].put(worker_request)


def start_generate_story(system_id, topic, storyline, story_temp):
    """Ask a system to start generating a story."""
    worker_request = {
        "action": "generate_story",
        "topic": topic,
        "storyline": storyline,
        "story_temp": story_temp
    }
    request_queues[system_id].put(worker_request)


def system_worker(system_id, story_generator_class, request_queue, result_queue):
        story_generator = story_generator_class(system_id)

        while True:
            worker_request = request_queue.get()
            action = worker_request["action"]
            if action == "generate":
                result = story_generator.generate_response(worker_request["topic"],
                                                           kw_temp=worker_request["kw_temp"],
                                                           story_temp=worker_request["story_temp"],
                                                           dedup=worker_request.get("dedup", None),
                                                           max_len=worker_request.get("max_len", None),
                                                           use_gold_titles=worker_request.get("use_gold_titles", None)
                )
            elif action == "generate_storyline":
                result = story_generator.generate_storyline(worker_request["topic"],
                                                            kw_temp=worker_request["kw_temp"],
                                                            dedup=worker_request.get("dedup", None),
                                                            max_len=worker_request.get("max_len", None),
                                                            use_gold_titles=worker_request.get("use_gold_titles", None)
                )
            elif action == "collab_storyline":
                result = story_generator.collab_storyline(worker_request["topic"], worker_request["storyline"],
                                                          kw_temp=worker_request.get("kw_temp", None),
                                                          dedup=worker_request.get("dedup", None),
                                                          max_len=worker_request.get("max_len", None))
            elif action == "generate_interactive_story":
                result = story_generator.generate_interactive_story(worker_request["topic"], worker_request["storyline"],
                                                        worker_request["story_sentences"], story_temp=worker_request.get("story_temp", None),
                                                                    only_one=worker_request["only_one"])
            elif action == "generate_story":
                result = story_generator.generate_story(worker_request["topic"], worker_request["storyline"],
                                                        story_temp=worker_request.get("story_temp", None))
            else:
                result = {
                    system_id: "internal error"
                }



            result_queue.put(result)


def get_response(system_id):
    """Returns a system's response as HTML."""
    response = result_queues[system_id].get()
    return response


if __name__ == '__main__':
    # By default, start an externally visible server (host='0.0.0.0')
    # on port 5000.  Set the environment variable CWC_SERVER_PORT to
    # change this.  For example, under bash or sh, you can start a
    # copy of the server on port 5001 with the following command:
    #
    # CWC_SERVER_PORT=5001 ./web_server.py
    #
    # You can also select the port by passing the "--port <port>"
    # option on the command line.  The command line --port option
    # overrides the CWC_SERVER_PORT environment variable.
    #
    # Similarly, the default host is "0.0.0.0", which means run the
    # server on all IP addresses available to the process.  This may
    # be changed with the CWC_SERVER_HOST environmen variable or the
    # "--host <host>" command line option.  One useful choice is
    # 127.0.0.1 (or "localhost"), which mmakes the server accessible
    # only to Web browsers running on the same system as the Web
    # server.
    #
    # 14-Sep-2018: On cwc-story.isi.edu, port number 80 has been
    # redirected to port 5000.  This allows an unprivileged user to
    # run the CWC Web server and make it visible on the default Web
    # port.
    default_host = os.environ.get("CWC_SERVER_HOST", "0.0.0.0")
    default_port = int(os.environ.get("CWC_SERVER_PORT", "5006"))

    parser = argparse.ArgumentParser()
    parser.add_argument(      '--debug', help="Run in debug mode (less restrictive).", required=False, action='store_true')
    parser.add_argument(      '--download-nltk', help="Download the NLTK data.", required=False, action='store_true')
    parser.add_argument(      '--host', help="The Web server host name or IP address", required=False, default=default_host)
    parser.add_argument('-p', '--port', help="The Web server port number (5000..5009)", required=False, type=int, default=default_port)
    args=parser.parse_args()

    if args.debug:
        # Force static files to timeout quickly to ease debugging.
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
        server_debug_mode = True

    print("Initializing the sentence generators.")
    initialize_generators()

    if args.download_nltk:
        nltk_resource_path = os.environ.get("CWC_NLTK_RESOURCE_PATH", "../nltk-data")
        print("Downloading the NLTK data from: %s" % (nltk_resource_path))
        nltk.download("wordnet", download_dir=nltk_resource_path)

    print("Starting the server on host %s port %d" % (args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True)
