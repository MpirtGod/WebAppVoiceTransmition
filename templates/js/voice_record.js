const recordButton = document.getElementById('record');
var class_timer = document.getElementById('timer');
var sec = 0;
var min = 0;
var hrs = 0;
var t;
var requestId;
let url;

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
  .then(function(stream) {
    var audioContext = new AudioContext();
    var sourceNode = audioContext.createMediaStreamSource(stream);
    var analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    sourceNode.connect(analyserNode);
    // analyserNode.connect(audioContext.destination);
    var dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    var canvas = document.getElementById('canvas');
    var canvasCtx = canvas.getContext('2d');
    function draw() {
      // Подготавливаем canvas для очередного кадра
      requestId = requestAnimationFrame(draw);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      // Получаем данные о звуке
      analyserNode.getByteFrequencyData(dataArray);
      // Рисуем столбцы визуализации
      var barWidth = canvas.width / 150;
      var barHeight;
      var x = 0;
      for (var i = 0; i < dataArray.length; i++) {
        barHeight = dataArray[i] / 2;
        canvasCtx.fillStyle = 'rgb(' + (barHeight + 100) + ',50,50)';
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth;
      }
    }
    // Запускаем визуализацию
    // Создаем MediaRecorder для записи звука
    var mediaRecorder = new MediaRecorder(stream);
    // Создаем массив для хранения записанного звука
    var chunks = [];
    // Начинаем запись при нажатии на кнопку
    document.getElementById('record').addEventListener('click', function() {
      if (mediaRecorder.state === 'inactive') {
                timer();
                mediaRecorder.start();
                recordButton.textContent = 'Остановить запись ⏹️';
                draw();
            } else {
                mediaRecorder.stop();
                chunks = []
                clearTimeout(t);
                class_timer.textContent = "00:00:00";
                sec = 0; min = 0; hrs = 0;
                recordButton.textContent = 'Записать голос ▶️';
                cancelAnimationFrame(requestId)
                canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
            }
    });

    mediaRecorder.addEventListener('dataavailable', function(event) {
      chunks.push(event.data);
    });

    mediaRecorder.addEventListener('stop', function() {
      // Создаем Blob из массива записанных данных
      var blob = new Blob(chunks, { type: 'audio/wav' });
      // Создаем ссылку на Blob
      url = URL.createObjectURL(blob);
      // Воспроизводим записанный звук
    });

    document.getElementById('play').addEventListener('click', function() {
      var audio = new Audio();
      // audio.addEventListener('play', function() {
      //     draw()
      // });
      //
      // audio.addEventListener('ended', function() {
      //     cancelAnimationFrame(requestId);
      //     canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      // });
      audio.src = url;
      audio.play();
    });
    // Обрабатываем данные после остановки записи
  })
  .catch(function(error) {
    console.log('Error:', error);
  });
