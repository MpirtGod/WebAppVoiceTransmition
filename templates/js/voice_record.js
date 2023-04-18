const recordButton = document.getElementById('recordButton');
let mediaRecorder;
let chunks = [];
let url;
let isPlaying = false;
var class_timer = document.getElementById('timer');
var sec = 0;
var min = 0;
var hrs = 0;
var t;

function tick(){
    sec++;
    if (sec >= 60) {
        sec = 0;
        min++;
        if (min >= 60) {
            min = 0;
            hrs++;
        }
    }
}

function add() {
    tick();
    class_timer.textContent = (hrs > 9 ? hrs : "0" + hrs)
        	 + ":" + (min > 9 ? min : "0" + min)
       		 + ":" + (sec > 9 ? sec : "0" + sec);
    timer();
}

function timer() {
    t = setTimeout(add, 1000);
}


navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        recordButton.addEventListener('click', () => {
            if (mediaRecorder.state === 'inactive') {
                timer();
                mediaRecorder.start();
                recordButton.textContent = 'Остановить запись ⏹️';
            } else {
                mediaRecorder.stop();
                clearTimeout(t);
                class_timer.textContent = "00:00:00";
                sec = 0; min = 0; hrs = 0;
                recordButton.textContent = 'Записать голос ▶️';
            }
        });

        mediaRecorder.addEventListener('dataavailable', event => {
            chunks.push(event.data);
        });

        mediaRecorder.addEventListener('stop', () => {
            const blob = new Blob(chunks, { type: 'audio/wav' });
            url = URL.createObjectURL(blob);
            chunks = [];

            // const a = document.createElement('a');

        document.getElementById('play-button').addEventListener('click', function() {
            var audio = new Audio();
            audio.src = url;
            audio.play();
            });

            // var xhr = new XMLHttpRequest();
            // xhr.open('POST', '/upload-audio', true);
            // xhr.setRequestHeader('Content-Type', 'application/octet-stream');
            // xhr.onreadystatechange = function() {
            //     if (xhr.readyState === 4 && xhr.status === 200) {
            //         console.log('Audio uploaded successfully');
            //     }
            // };
            // xhr.send(blob);

            // a.href = url;
            // a.download = 'recording.wav';
            // document.body.appendChild(a);
            // a.click();
          });
        })
        .catch(error => {
            console.error(error);
        });