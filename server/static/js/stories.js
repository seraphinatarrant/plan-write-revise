$('#translate-button').click(function() {
    var btn = $(this);
    // btn.button('loading');
    $("#status").html("Generating...");
    $("#system_1").html("");
    $("#system_2").html("");
    $("#system_3").html("");
    $("#elapsed").html("");

    $("#system_1_ready").html("");
    $("#system_2_ready").html("");
    $("#system_3_ready").html("");

    var topic = $('#input-topic').val();
    // topic = topic.replace(/ /g, "_");
    if (topic == ""){
	topic = "the not so haunted house";
    }
    var kw_temp = $('#kw_temp').val();
    var story_temp = $('#story_temp').val();
    var system_id = "system_2";
    var dedup = $('#dedup').is(":checked");
    var max_len = $('#max_len').val();
    var use_gold_titles = $('#use_gold').is(":checked");
    var debug_mode = $('#debug_mode').is(":checked");
    var system1_story = "";
    var system2_story = "";
    var system3_story = "";
    if (!debug_mode) {
	$("#translate-button").prop("disabled", true);
    }

    id = Math.round(Math.random()*10000) + 1;
    $.ajax({
	type:"GET",
	url: "/api/generate_storyline",
	data: {
	    id: id,
	    topic: topic,
	    systems: "system_2",
	    kw_temp: kw_temp,
	    dedup: dedup,
	    max_len: max_len,
	    use_gold_titles: use_gold_titles
	},
	xhrFields: {
	    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    // This can be used to set the 'withCredentials' property.
	    // Set the value to 'true' if you'd like to pass cookies to the server.
	    // If this is enabled, your server must respond with the header
	    // 'Access-Control-Allow-Credentials: true'.
	    withCredentials: false
	},
	success: function(response_data) {

	    // The response data is sent as "application/json",
	    // and is automatically decoded.
	    // jd = $.parseJSON(response_data);
	    jd = response_data;
	    $("#elapsed").html(jd.elapsed + " seconds");
	    var storyline = jd.system_2
	    $("#storyline").text(storyline);

	    $.ajax({
		type:"GET",
		url: "/api/generate",
		data: {
		    id: id,
		    systems: "system_1",
		    topic: topic,
		    storyline: storyline,
		    story_temp: story_temp,
		    dedup: dedup,
		    max_len: max_len
		},
		xhrFields: {
		    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
		    // This can be used to set the 'withCredentials' property.
		    // Set the value to 'true' if you'd like to pass cookies to the server.
		    // If this is enabled, your server must respond with the header
		    // 'Access-Control-Allow-Credentials: true'.
		    withCredentials: false
		},
		success: function(response_data) {

		    // The response data is sent as "application/json",
		    // and is automatically decoded.
		    // jd = $.parseJSON(response_data);
		    jd = response_data;
		    system1_story = jd.system_1;
		    $("#system_1").html(jd.system_1);
		    $("#nstory").html(jd.n_story);
		    $("#story_id").html(jd.story_id);
		    $("#elapsed").html(jd.elapsed + " seconds");
		    $("#system_1_ready").html("Ready");
		    if ($("#system_1_ready").html() == "Ready" &&
			$("#system_2_ready").html() == "Ready" &&
			$("#system_3_ready").html() == "Ready") {
			$("#status").html("Ready");
			$("#translate-button").prop("disabled", false);
		    }
		}
	    });
	    $.ajax({
		type:"GET",
		url: "/api/generate_story",
		data: {
		    id: id,
		    system_id: "system_2",
		    topic: topic,
		    storyline: storyline,
		    story_temp: story_temp
		},
		xhrFields: {
		    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
		    // This can be used to set the 'withCredentials' property.
		    // Set the value to 'true' if you'd like to pass cookies to the server.
		    // If this is enabled, your server must respond with the header
		    // 'Access-Control-Allow-Credentials: true'.
		    withCredentials: false
		},
		success: function(response_data) {

		    // The response data is sent as "application/json",
		    // and is automatically decoded.
		    // jd = $.parseJSON(response_data);
		    jd = response_data;
		    system2_story = jd.story;
		    $("#system_2").html(jd.story);
		    $("#nstory").html(jd.n_story);
		    $("#story_id").html(jd.story_id);
		    $("#elapsed").html(jd.elapsed + " seconds");
		    $("#system_2_ready").html("Ready");
		    if ($("#system_1_ready").html() == "Ready" &&
			$("#system_2_ready").html() == "Ready" &&
			$("#system_3_ready").html() == "Ready") {
			$("#status").html("Ready");
			$("#translate-button").prop("disabled", false);
		    }
		}
	    });
	    $.ajax({
		type:"GET",
		url: "/api/generate_story",
		data: {
		    id: id,
		    system_id: "system_3",
		    topic: topic,
		    storyline: storyline,
		    story_temp: story_temp
		},
		xhrFields: {
		    // The 'xhrFields' property sets additional fields on the XMLHttpRequest.
		    // This can be used to set the 'withCredentials' property.
		    // Set the value to 'true' if you'd like to pass cookies to the server.
		    // If this is enabled, your server must respond with the header
		    // 'Access-Control-Allow-Credentials: true'.
		    withCredentials: false
		},
		success: function(response_data) {

		    // The response data is sent as "application/json",
		    // and is automatically decoded.
		    // jd = $.parseJSON(response_data);
		    jd = response_data;
		    system3_story = jd.story;
		    $("#system_3").html(jd.story);
		    $("#nstory").html(jd.n_story);
		    $("#story_id").html(jd.story_id);
		    $("#elapsed").html(jd.elapsed + " seconds");
		    $("#system_3_ready").html("Ready");
		    if ($("#system_1_ready").html() == "Ready" &&
			$("#system_2_ready").html() == "Ready" &&
			$("#system_3_ready").html() == "Ready") {
			$("#status").html("Ready");
			$("#translate-button").prop("disabled", false);
		    }
		}
	    });
	    var checkData = setInterval(check, 3000);
    	function check() {
			if (system1_story.length != 0 && system2_story.length != 0 && system3_story.length != 0) {
				$.ajax({
					type:"POST",
					url: "/api/write_auto_txt",
					data: JSON.stringify({
						topic: topic,
						storyline: storyline,
						story_temp: story_temp,
						system1_story: system1_story,
						system2_story: system2_story,
						system3_story: system3_story

					}),
					xhrFields: {
	    				// The 'xhrFields' property sets additional fields on the XMLHttpRequest.
	    				// This can be used to set the 'withCredentials' property.
	    				// Set the value to 'true' if you'd like to pass cookies to the server.
	    				// If this is enabled, your server must respond with the header
	    				// 'Access-Control-Allow-Credentials: true'.
	    				withCredentials: false
					},
					success: function(response_data) {
	    				console.log("success write auto logging txt");
					}

				});
				clearInterval(checkData);
			}
		}


	}
    });

});
