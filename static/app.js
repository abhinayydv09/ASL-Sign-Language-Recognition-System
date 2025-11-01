const synthesis = window.speechSynthesis;
const recognition = new webkitSpeechRecognition();
recognition.continuous = true;
recognition.lang = 'en-US';

recognition.onresult = function (event) {
    const transcript = event.results[event.results.length - 1][0].transcript;
    document.getElementById('output').innerText = transcript;

    speakText(transcript);
};

recognition.onerror = function (event) {
    console.error('Speech recognition error:', event.error);
};

recognition.onend = function () {
    recognition.start();
};
function startVideo() {
    console.log('Starting video...');
    recognition.start();
    const videoFeed = document.getElementById('video-feed');
    const videoFeedUrl = "{{ url_for('video_feed') }}";
    console.log('Video Feed URL:', videoFeedUrl);
    videoFeed.src = videoFeedUrl;
    videoFeed.style.border = "4px solid #4CAF50";  // Add border as an indicator
}

function stopVideo() {
    console.log('Stopping video...');
    recognition.stop();
    const videoFeed = document.getElementById('video-feed');
    videoFeed.src = "";
    videoFeed.style.border = "none";  // Remove border
}

function speakText(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.volume = document.getElementById('volume').value;
    synthesis.speak(utterance);
}

function changeVolume() {
    document.getElementById('output').innerText = '';  // Clear output when volume changes
    document.getElementById('volume').value;
}
