$(document).ready(function() {
    $('[data-toggle="tooltip"]').tooltip;
});

// record the original story
var story_sentences = ["", "", "", "", ""];

$('#start-button').click(function() {
    var btn = $(this);
    var topic = $('#input-topic').val();
    if (topic == "") {
	alert("Please enter a title.")
	return;
    }

    $("#input-topic").prop("readonly", true);
    $("#start-button").prop("disabled", true);
    $("#clear-button").prop("disabled", false);
    $("#suggest-button").prop("disabled", false);
    $("#suggest-story-button").prop("disabled", false);
    $("#clear-story-button").prop("disabled", false);
    $("#generate-story-button").prop("disabled", false);
    $("#storyline_phrase_1").prop("readonly", false);
    $("#storyline_phrase_2").prop("readonly", false);
    $("#storyline_phrase_3").prop("readonly", false);
    $("#storyline_phrase_4").prop("readonly", false);
    $("#storyline_phrase_5").prop("readonly", false);
    $("#story_sentence_1").prop("readonly", false);
    $("#story_sentence_2").prop("readonly", false);
    $("#story_sentence_3").prop("readonly", false);
    $("#story_sentence_4").prop("readonly", false);
    $("#story_sentence_5").prop("readonly", false);
    $("#storyline_phrase_1").val("");
    $("#storyline_phrase_2").val("");
    $("#storyline_phrase_3").val("");
    $("#storyline_phrase_4").val("");
    $("#storyline_phrase_5").val("");
    $("#story_sentence_1").val("");
    $("#story_sentence_2").val("");
    $("#story_sentence_3").val("");
    $("#story_sentence_4").val("");
    $("#story_sentence_5").val("");
    $("#you-may-edit").removeClass("invisible").addClass("show");
    $("#you-may-edit-story").removeClass("invisible").addClass("show");
    $("#elapsed").html("");
});
			 
$('#reset-button').click(function() {
    var btn = $(this);
    story_sentences = ["", "", "", "", ""];
    $("#input-topic").prop("readonly", false);
    $("#input-topic").val("");
    $("#start-button").prop("disabled", false);
    $("#clear-button").prop("disabled", true);
    $("#suggest-button").prop("disabled", true);
    $("#suggest-story-button").prop("disabled", true);
    $("#clear-story-button").prop("disabled", true);
    $("#generate-story-button").prop("disabled", true);
    $("#storyline_phrase_1").prop("readonly", true);
    $("#storyline_phrase_2").prop("readonly", true);
    $("#storyline_phrase_3").prop("readonly", true);
    $("#storyline_phrase_4").prop("readonly", true);
    $("#storyline_phrase_5").prop("readonly", true);
    $("#story_sentence_1").prop("readonly", true);
    $("#story_sentence_2").prop("readonly", true);
    $("#story_sentence_3").prop("readonly", true);
    $("#story_sentence_4").prop("readonly", true);
    $("#story_sentence_5").prop("readonly", true);
    $("#storyline_phrase_1").val("");
    $("#storyline_phrase_2").val("");
    $("#storyline_phrase_3").val("");
    $("#storyline_phrase_4").val("");
    $("#storyline_phrase_5").val("");
    $("#story_sentence_1").val("");
    $("#story_sentence_2").val("");
    $("#story_sentence_3").val("");
    $("#story_sentence_4").val("");
    $("#story_sentence_5").val("");
    $("#you-may-edit").removeClass("show").addClass("invisible");
    $("#you-may-edit-story").removeClass("show").addClass("invisible");
    $("#elapsed").html("");
});
			 
$('#clear-button').click(function() {
    var btn = $(this);
    story_sentences = ["", "", "", "", ""];
    $("#storyline_phrase_1").val("");
    $("#storyline_phrase_2").val("");
    $("#storyline_phrase_3").val("");
    $("#storyline_phrase_4").val("");
    $("#storyline_phrase_5").val("");
    $("#elapsed").html("");
});

$('#clear-story-button').click(function() {
    var btn = $(this);
    story_sentences = ["", "", "", "", ""];
    $("#story_sentence_1").val("");
    $("#story_sentence_2").val("");
    $("#story_sentence_3").val("");
    $("#story_sentence_4").val("");
    $("#story_sentence_5").val("");
    $("#elapsed").html("");
});
			 
function build_storyline() {
    var storyline = "";
    if ($('#storyline_phrase_1').val() == "") {
	return storyline;
    }
    storyline = $('#storyline_phrase_1').val();
    if ($('#storyline_phrase_2').val() == "") {
	return storyline;
    }
    storyline = storyline + " -> " + $('#storyline_phrase_2').val();
    if ($('#storyline_phrase_3').val() == "") {
	return storyline;
    }
    storyline = storyline + " -> " + $('#storyline_phrase_3').val();
    if ($('#storyline_phrase_4').val() == "") {
	return storyline;
    }
    storyline = storyline + " -> " + $('#storyline_phrase_4').val();
    if ($('#storyline_phrase_5').val() == "") {
	return storyline;
    }
    storyline = storyline + " -> " + $('#storyline_phrase_5').val();
    return storyline;
}

function extend_storyline(new_phrase) {
    // console.log("Suggesting " + new_phrase);
    if ($('#storyline_phrase_1').val() == "") {
	// console.log("for phrase 1");
	$('#storyline_phrase_1').val(new_phrase);
	return;
    }
    if ($('#storyline_phrase_2').val() == "") {
	// console.log("for phrase 2");
	$('#storyline_phrase_2').val(new_phrase);
	return;
    }
    if ($('#storyline_phrase_3').val() == "") {
	// console.log("for phrase 3");
	$('#storyline_phrase_3').val(new_phrase);
	return;
    }
    if ($('#storyline_phrase_4').val() == "") {
	// console.log("for phrase 4");
	$('#storyline_phrase_4').val(new_phrase);
	return;
    }
    if ($('#storyline_phrase_5').val() == "") {
	// console.log("for phrase 5");
	$('#storyline_phrase_5').val(new_phrase);
	return;
    }
    // console.log("... ignored, all 5 phrases have been filled");
}

function generate_story_sentences(story, only_one) {
    console.log("only one " + only_one); // bool for if someone clicks
    var sentences = story.split(" </s> "); // sentences is the story split on the sep char
    // should add error catching here so that if leads with a </s> then remove index 0 TODO
    if ($('#story_sentence_1').val() == "") {
	    $('#story_sentence_1').val(sentences[0]);
	    story_sentences[0] = sentences[0];
        if(only_one==true)return;
    }
    if ($('#story_sentence_2').val() == "") {
	    $('#story_sentence_2').val(sentences[1]);
	    story_sentences[1] = sentences[1];
        if(only_one==true)return;
    }
    if ($('#story_sentence_3').val() == "") {
	    $('#story_sentence_3').val(sentences[2]);
	    story_sentences[2] = sentences[2];
        if(only_one==true)return;
    }
    if ($('#story_sentence_4').val() == "") {
	    $('#story_sentence_4').val(sentences[3]);
	    story_sentences[3] = sentences[3];
        if(only_one==true)return;
    }
    if ($('#story_sentence_5').val() == "") {
	    $('#story_sentence_5').val(sentences[4]);
	    story_sentences[4] = sentences[4];
        if(only_one==true)return;
    }

}

function extend_storyline_from_response(response_data) {
    extend_storyline(response_data.new_phrase)
}

function get_topic() {
    var topic = $('#input-topic').val();
    // topic = topic.replace(/ /g, "_");
    if (topic == ""){
	topic = "the not so haunted house";
    }
    return topic;
}



// generate data without sentences
function build_request_data() {
    var id = Math.round(Math.random()*10000) + 1;

    var system_id = "system_2";
    if ($('#system_3').is(":checked")) {
	system_id = "system_3"
    }

    var topic = get_topic();
    var storyline = build_storyline();

    var kw_temp = $('#kw_temp').val();
    var story_temp = $('#story_temp').val();
    var dedup = $('#dedup').is(":checked");
    var max_len = $('#max_len').val();

    return {
	id: id,
	system_id: system_id,
	topic: topic,
	storyline: storyline,
	kw_temp: kw_temp,
	story_temp: story_temp,
	dedup: dedup,
	max_len: max_len
    };
}


// generate data with sentences
function build_request_story_data() {
    console.log("current story", story_sentences);
    var id = Math.round(Math.random()*10000) + 1;

    var system_id = "system_2";
    if ($('#system_3').is(":checked")) {
	system_id = "system_3"
    }

    var topic = get_topic();
    var storyline = build_storyline();

    var kw_temp = $('#kw_temp').val();
    var story_temp = $('#story_temp').val();
    var dedup = $('#dedup').is(":checked");
    var max_len = $('#max_len').val();

    // if use sentence as input
    var use_sentence = false;
    var sentences = ["", "", "", "", ""];
    var change_max_idx = -1;
    for(var i = 0; i < 5; i++) {
        // sentences are 1 indexed. So for every sentence,
        // check whether the there is content in the UI that isn't matching the global story sentences array
        // and if it isn't, then change_max_idx becomes i. So if anything is cleared or changed,
        // this number will be the maximum of the one that was
        if (story_sentences[i] != $('#story_sentence_' + (i+1)).val()) {
            console.log($('#story_sentence_' + (i+1)).val());
            use_sentence = true;
            change_max_idx = Math.max(change_max_idx, i);
        }
        //local array sentences will be filled with whatever is in the UI
        sentences[i] = $('#story_sentence_' + (i+1)).val();
        console.log("sentences[i] is " + sentences[i]);
        console.log("change max idx is " + change_max_idx);
    }

    if (!use_sentence) {
        for (var i = 0; i < 5; i++) {
            if (sentences[i] == "") {
                use_sentence = true;
                change_max_idx = i;
                break;
            }
        }
    }

    return {
	    id: id,
	    system_id: system_id,
	    topic: topic,
	    storyline: storyline,
	    kw_temp: kw_temp,
	    story_temp: story_temp,
	    dedup: dedup,
	    max_len: max_len,
        use_sentence: use_sentence,
        sentences: sentences,
        change_max_idx: change_max_idx
    };
}

$('#suggest-button').click(function() {
    var btn = $(this);
    // btn.button('loading');
    $("#status").html("Starting...");
    // $("#story").html("");
    $("#elapsed").html("");
    // $('#extend-button').prop("disabled", false);

    $.ajax({
	type:"GET",
	url: "/api/collab_storyline",
	data: build_request_data(),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    extend_storyline_from_response(response_data)
	    $("#elapsed").html(response_data.elapsed + " seconds");
	    $("#status").html("Ready");
	}
    });

});

$('#generate-story-button').click(function() {
    var btn = $(this);
    // btn.button('loading');
    $("#status").html("Generating...");
    // $("#story").html("");
    $("#elapsed").html("");

    complete_phrase_1();
});

$('#suggest-story-button').click(function() {
    var btn = $(this);
    // btn.button('loading');
    $("#status").html("Generating...");
    // $("#story").html("");
    $("#elapsed").html("");

    complete_phrase_1(true);
});

function complete_phrase_1(only_one=false) {
    if ($('#storyline_phrase_1').val() != "") {
	complete_phrase_2(only_one);
	return;
    }	
    $.ajax({
	type:"GET",
	url: "/api/collab_storyline",
	data: build_request_data(),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    extend_storyline_from_response(response_data)
	    complete_phrase_2(only_one);
	}
    });
}

function complete_phrase_2(only_one=false) {
    if ($('#storyline_phrase_2').val() != "") {
	complete_phrase_3(only_one);
	return;
    }	
    $.ajax({
	type:"GET",
	url: "/api/collab_storyline",
	data: build_request_data(),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    extend_storyline_from_response(response_data)
	    complete_phrase_3(only_one);
	}
    });
}

function complete_phrase_3(only_one=false) {
    if ($('#storyline_phrase_3').val() != "") {
	complete_phrase_4(only_one);
	return;
    }	
    $.ajax({
	type:"GET",
	url: "/api/collab_storyline",
	data: build_request_data(),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    extend_storyline_from_response(response_data)
	    complete_phrase_4(only_one);
	}
    });
}

function complete_phrase_4(only_one=false) {
    if ($('#storyline_phrase_4').val() != "") {
	complete_phrase_5(only_one);
	return;
    }	
    $.ajax({
	type:"GET",
	url: "/api/collab_storyline",
	data: build_request_data(),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    extend_storyline_from_response(response_data)
	    complete_phrase_5(only_one);
	}
    });
}

function complete_phrase_5(only_one=false) {
    if ($('#storyline_phrase_5').val() != "") {
	generate_story(only_one);
	return;
    }	
    $.ajax({
	type:"GET",
	url: "/api/collab_storyline",
	data: build_request_data(),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    extend_storyline_from_response(response_data)
	    generate_story(only_one);
	}
    });
}

function generate_story(only_one=false) {
    var story_data = build_request_story_data();
    // starrant added this, it's not the cleanest
    story_data['only_one'] = only_one
    $.ajax({
	type:"POST",
	url: "/api/generate_interactive_story",
	data: JSON.stringify(story_data),
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {
	    generate_story_sentences(response_data.story,only_one);
	    $("#elapsed").html(response_data.elapsed + " seconds");
	    $("#status").html("Ready");
	    if ($('#story_sentence_1').val() != '' && $('#story_sentence_2').val() != '' &&
            $('#story_sentence_3').val() != '' && $('#story_sentence_4').val() != '' &&
            $('#story_sentence_5').val() != '') {
	        $.ajax({
                type:"POST",
	            url: "/api/write_interactive_txt",
                data: JSON.stringify({
                    topic: story_data.topic,
	                storyline: story_data.storyline,
	                kw_temp: story_data.kw_temp,
	                story_temp: story_data.story_temp,
                    story: response_data.story
                }),
                xhrFields: {
	                // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	                // This can be used to set the 'withCredentials' property.
	                // Set the value to 'true' if you'd like to pass cookies to the server.
	                // If this is enabled, your server must respond with the header
                    // 'Access-Control-Allow-Credentials: true'.
	                withCredentials: false
	            },
                success: function(response) {
                    console.log("success write interactive logging txt");
                }
            });
        }

	}
    });
}

