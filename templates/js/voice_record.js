const recordButton = document.getElementById('recordButton');
      let mediaRecorder;
      let chunks = [];

      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          mediaRecorder = new MediaRecorder(stream);

          recordButton.addEventListener('click', () => {
            if (mediaRecorder.state === 'inactive') {
              mediaRecorder.start();
              recordButton.textContent = 'Остановить запись ⏹️';
            } else {
              mediaRecorder.stop();
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