const recordButton = document.getElementById('record');
const stopButton = document.getElementById('stop');
const playButton = document.getElementById('play');

const timer = document.getElementById('timer');
let interval;
let seconds = 0, minutes = 0, hours = 0;

const canvas = document.getElementById("canvas");
let canvasCtx = canvas.getContext("2d");

let analyserNode;
let sourceNode;
let requestId;

let url;
let audio;
let pause;

function startTimer() {
  interval = setInterval(function() {
   seconds++;
   if (seconds == 60) {
    seconds = 0;
    minutes++;
    if (minutes == 60) {
     minutes = 0;
     hours++;
    }
   }
   timer.innerHTML = (hours ? (hours > 9 ? hours : "0" + hours) : "00") + ":" + (minutes ? (minutes > 9 ? minutes : "0" + minutes) : "00") + ":" + (seconds > 9 ? seconds : "0" + seconds);
  }, 1000);
}

function stopTimer() {
  clearInterval(interval);
}

function resetTimer() {
  clearInterval(interval);
  seconds = 0;
  minutes = 0;
  hours = 0;
  timer.innerHTML = "00:00:00";
}

function draw() {
  analyserNode.fftSize = 2048;
  sourceNode.connect(analyserNode);
  let dataArray = new Uint8Array(analyserNode.frequencyBinCount);
  requestId = requestAnimationFrame(draw);
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  analyserNode.getByteFrequencyData(dataArray);
  let barWidth = canvas.width / 150;
  let barHeight;
  let x = 0;
  for (let i = 0; i < dataArray.length; i++) {
    barHeight = dataArray[i] / 2;
    if (barHeight < 1) { // Если нет звука, рисуем один пиксель
      barHeight = 1;
    }
    canvasCtx.fillStyle = `rgb(${barHeight * 1.3},175,80)`;
    canvasCtx.fillRect(x, (canvas.height - barHeight) / 2, barWidth, barHeight);
    x += barWidth;
  }
}

function sendVoice(form) {
  let request = new XMLHttpRequest();
  request.overrideMimeType('text/plain');

  request.onload = function() {
    if (request.status != 200) {
      alert("Не приняли!");
    } else {
      alert(request.responseText);
    }
  }

  request.open("POST", "/upload-audio");
  request.send(form);
}

async function getMedia(constraints) {
  let stream = null;

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    console.log("Микрофон инициализирован!!!");
  } catch(err) {
    alert("Ошибка инициализации записывающего устройства!");
  }

  if (stream == null) {
    return;
  }

  let audioContext = new AudioContext();
  let mediaRecorder = new MediaRecorder(stream);
  let chunks = [];

  recordButton.addEventListener('click', function() {
    if (mediaRecorder.state != 'inactive') {
      recordButton.style.backgroundColor = "#FF0000";
      recordButton.textContent = "Запись";
      mediaRecorder.stop();
      stopTimer();
      cancelAnimationFrame(requestId);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    recordButton.style.backgroundColor = "#0000FF";
    recordButton.textContent = "Остановить";
    resetTimer();
    startTimer();
    mediaRecorder.start();
    sourceNode = audioContext.createMediaStreamSource(stream);
    analyserNode = audioContext.createAnalyser();
    draw();
  });

  mediaRecorder.addEventListener('dataavailable', function(event) {
    chunks.push(event.data);
  });

  mediaRecorder.addEventListener('stop', function() {
    let blob = new Blob(chunks, { type: 'audio/wav' });
    let fd = new FormData();
    fd.append('voice', blob);
    sendVoice(fd);
    url = URL.createObjectURL(blob);
    chunks = [];
  });

  playButton.addEventListener('click', function() {
    if (url === undefined) {
      return;
    }

    if (audio !== undefined && !pause) {
      pause = true;
      audio.pause()
      return
    }

    if (audio !== undefined && pause) {
      pause = false;
      audio.play()
      return
    }

    audio = new Audio();
    pause = false;

    audio.addEventListener('play', function() {

    });
  });
}

getMedia({ audio: true });
