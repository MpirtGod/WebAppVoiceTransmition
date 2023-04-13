const recordButton = document.getElementById('recordButton');
let mediaRecorder;
let chunks = [];
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
                seconds = 0; minutes = 0; hours = 0;
                recordButton.textContent = 'Записать голос ▶️';
            }
        });

        mediaRecorder.addEventListener('dataavailable', event => {
            chunks.push(event.data);
        });

        mediaRecorder.addEventListener('stop', () => {
            const blob = new Blob(chunks, { type: 'audio/ogg; codecs=opus' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'recording.ogg';
            document.body.appendChild(a);
            a.click();
            chunks = [];
          });
        })
        .catch(error => {
            console.error(error);
        });