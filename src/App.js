import { useRef, useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

import Button from '@mui/material/Button';
import Webcam from 'react-webcam';


function App() {

  const [currentMusicDetails, setCurrentMusicDetails] = useState({
    songName: 'Chasing',
    songArtist: 'NEFFEX',
    songSrc: './Assets/songs/Chasing - NEFFEX.mp3',
    songAvatar: './Assets/Images/image1.jpg'
  })

  //UseStates Variables
  const [audioProgress, setAudioProgress] = useState(0);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [musicIndex, setMusicIndex] = useState(0);
  const [musicTotalLength, setMusicTotalLength] = useState('04 : 38');
  const [musicCurrentTime, setMusicCurrentTime] = useState('00 : 00');
  const [videoIndex, setVideoIndex] = useState(0)

  const currentAudio = useRef()

  const handleMusicProgressBar = (e)=>{
    setAudioProgress(e.target.value);
    currentAudio.current.currentTime = e.target.value * currentAudio.current.duration / 100;
  }

  //Change Avatar Class
  let avatarClass = ['objectFitCover','objectFitContain','none']
  const [avatarClassIndex, setAvatarClassIndex] = useState(0)
  const handleAvatar = ()=>{
    if (avatarClassIndex >= avatarClass.length - 1) {
      setAvatarClassIndex(0)
    }else{
      setAvatarClassIndex(avatarClassIndex + 1)
    }
  }


  //Play Audio Function
  const handleAudioPlay = ()=>{
    if (currentAudio.current.paused) {
      currentAudio.current.play();
      setIsAudioPlaying(true)
    }else{
      currentAudio.current.pause();
      setIsAudioPlaying(false)
    }
  }

  const musicAPI = [
    {
      songName: 'Chasing',
      songArtist: 'NEFFEX',
      songSrc: './Assets/songs/Chasing - NEFFEX.mp3',
      songAvatar: './Assets/Images/image1.jpg'
    },
    {
      songName: 'AURORA - Runaway',
      songArtist: 'Aurora Aksnes',
      songSrc: './Assets/songs/AURORA - Runaway (Lyrics).mp3',
      songAvatar: './Assets/Images/image4.jpg'
    },
    {
      songName: 'Catch Me If I Fall',
      songArtist: 'TEGNENT',
      songSrc: './Assets/songs/Catch Me If I Fall - NEFFEX.mp3',
      songAvatar: './Assets/Images/image2.jpg'
    },
    {
      songName: 'Inspired (Clean)',
      songArtist: 'NEFFEX',
      songSrc: './Assets/songs/Inspired (Clean) - NEFFEX.mp3',
      songAvatar: './Assets/Images/image3.jpg'
    },
    {
      songName: 'Baby doll [ slowed + reverb ]',
      songArtist: 'Kanika Kapoor',
      songSrc: './Assets/songs/Baby doll [ slowed + reverb ] __ meet bros ,Kanika Kapoor __ jr santu.mp3',
      songAvatar: './Assets/Images/image5.jpg'
    },
    {
      songName: 'Soch (Slowed+Reverbed)',
      songArtist: 'Hardy Sandhu',
      songSrc: './Assets/songs/SOCH(Slowed+Reverbed) __ Hardy Sandhu.webm',
      songAvatar: './Assets/Images/image6.jpg'
    },
    {
      songName: 'Apna Bana Le',
      songArtist: 'Arijit Singh',
      songSrc: './Assets/songs/Apna Bana Le - Full Audio _ Bhediya _ Varun Dhawan, Kriti Sanon_ Sachin-Jigar,Arijit Singh,Amitabh B.webm',
      songAvatar: './Assets/Images/image7.jpg'
    },
    {
      songName: 'Arcade',
      songArtist: 'Duncan Laurence',
      songSrc: './Assets/songs/Arcade - Duncan Laurence.m4a',
      songAvatar: './Assets/Images/image7.jpg'
    },
    {
      songName: 'Give Me a Kiss',
      songArtist: 'Crash Adams',
      songSrc: './Assets/songs/Give Me a Kiss - Crash Adams.m4a',
      songAvatar: './Assets/Images/image7.jpg'
    },
    {
      songName: 'Make It Out Alive',
      songArtist: 'Malachiii',
      songSrc: './Assets/songs/Make_It_Out_Alive_The_Spider_Within_A_Spider_Verse_Story_Malachiii.m4a',
      songAvatar: './Assets/Images/image7.jpg'
    },
    {
      songName: 'Somebody',
      songArtist: 'The Chainsmokers, Drew Love',
      songSrc: './Assets/songs/Somebody - The Chainsmokers, Drew Love.m4a',
      songAvatar: './Assets/Images/image7.jpg'
    },
    {
      songName: 'Viva La Vida',
      songArtist: 'Coldplay',
      songSrc: './Assets/songs/Viva La Vida - Coldplay.m4a',
      songAvatar: './Assets/Images/image7.jpg'
    }
  ]

  const handleNextSong = ()=>{
    if (musicIndex >= musicAPI.length - 1) {
      let setNumber = 0;
      setMusicIndex(setNumber);
      updateCurrentMusicDetails(setNumber);
    }else{
      let setNumber = musicIndex + 1;
      setMusicIndex(setNumber)
      updateCurrentMusicDetails(setNumber);
    }
  }

  const handlePrevSong = ()=>{
    if (musicIndex === 0) {
      let setNumber = musicAPI.length - 1;
      setMusicIndex(setNumber);
      updateCurrentMusicDetails(setNumber);
    }else{
      let setNumber = musicIndex - 1;
      setMusicIndex(setNumber)
      updateCurrentMusicDetails(setNumber);
    }
  }

  const updateCurrentMusicDetails = (number)=>{
    let musicObject = musicAPI[number];
    currentAudio.current.src = musicObject.songSrc;
    currentAudio.current.play();
    setCurrentMusicDetails({
      songName: musicObject.songName,
      songArtist: musicObject.songArtist,
      songSrc: musicObject.songSrc,
      songAvatar: musicObject.songAvatar
    })
    setIsAudioPlaying(true);
  }

  const handleAudioUpdate = ()=>{
    //Input total length of the audio
    let minutes = Math.floor(currentAudio.current.duration / 60);
    let seconds = Math.floor(currentAudio.current.duration % 60);
    let musicTotalLength0 = `${minutes <10 ? `0${minutes}` : minutes} : ${seconds <10 ? `0${seconds}` : seconds}`;
    setMusicTotalLength(musicTotalLength0);

    //Input Music Current Time
    let currentMin = Math.floor(currentAudio.current.currentTime / 60);
    let currentSec = Math.floor(currentAudio.current.currentTime % 60);
    let musicCurrentT = `${currentMin <10 ? `0${currentMin}` : currentMin} : ${currentSec <10 ? `0${currentSec}` : currentSec}`;
    setMusicCurrentTime(musicCurrentT);

    const progress = parseInt((currentAudio.current.currentTime / currentAudio.current.duration) * 100);
    setAudioProgress(isNaN(progress)? 0 : progress)
  }


  const vidArray = ['./Assets/Videos/video1.mp4','./Assets/Videos/video2.mp4','./Assets/Videos/video3.mp4','./Assets/Videos/video4.mp4','./Assets/Videos/video5.mp4','./Assets/Videos/video6.mp4'];

  const [camera, setCamera] = useState(false);
  const [lyrics, setLyrics] = useState(false);
  const [lyrics2, setLyrics2] = useState('');
  const [lyrics3, setLyrics3] = useState('');



  const toggleCamera = ()=>{
    console.log(camera);
    if (camera) {
      setCamera(false);
    }else{
      setCamera(true);
    }
  }

  const toggleLyrics = async () => {
    if (lyrics) {
      setLyrics(false);
    } else {
      setLyrics(true);
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/get_data/', {
          params: {
            format: 'json',
            q_artist: currentMusicDetails.songArtist,
            q_track: currentMusicDetails.songName,
            apikey: 'a4b507af272255aa1488b4eb10fd85d6'
          }
        });

        if (response.status === 200) {
          const newLyrics = response.data.message.body.lyrics.lyrics_body;
          setLyrics2(newLyrics); 
        } else {
          console.error('Error fetching lyrics:', response.status);
        }
      } catch (error) {
        console.error('Error fetching lyrics:', error);
      } finally {
        setLyrics(false);
      }
    }
  };

  return (
    <>
    <div className="container">
      <audio src='./Assets/songs/Chasing - NEFFEX.mp3' ref={currentAudio} onEnded={handleNextSong} onTimeUpdate={handleAudioUpdate}></audio>
      <video src={vidArray[videoIndex]} loop muted autoPlay className='backgroundVideo'></video>
      <div className="blackScreen"></div>

      {
      lyrics2 && 
        <div className="music-Container">
          <div>
          <p className='musicPlayer'>Lyrics</p>
          <p style={{ fontSize: '12px' }}>{lyrics2}</p>
          </div>
        </div>
      }
        
      <div className="music-Container">
        <p className='musicPlayer'>Music Player</p>
        <p className='music-Head-Name'>{currentMusicDetails.songName}</p>
        <p className='music-Artist-Name'>{currentMusicDetails.songArtist}</p>
        <img src={currentMusicDetails.songAvatar} className={avatarClass[avatarClassIndex]} onClick={handleAvatar} alt="song Avatar" id='songAvatar'/>
        <div className="musicTimerDiv">
          <p className='musicCurrentTime'>{musicCurrentTime}</p>
          <p className='musicTotalLenght'>{musicTotalLength}</p>
        </div>
        <input type="range" name="musicProgressBar" className='musicProgressBar' value={audioProgress} onChange={handleMusicProgressBar} />
        <div className="musicControlers">
          <i className='fa-solid fa-backward musicControler' onClick={handlePrevSong}></i>
          <i className={`fa-solid ${isAudioPlaying? 'fa-pause-circle' : 'fa-circle-play'} playBtn`} onClick={handleAudioPlay}></i>
          <i className='fa-solid fa-forward musicControler' onClick={handleNextSong}></i>
        </div>
      </div>

      {
      camera && 
        <div className="music-Container">
          <p className='musicPlayer'>Mood Detection</p>
          <Webcam width={300} />
        </div>
      }

      <div>
        <div className="changeBackBtn" onClick={toggleLyrics}>
          Show Lyrics
        </div>

        <div className="changeBackBtn2" onClick={toggleCamera}>
          Turn On Camera
        </div>
      </div>
      <a href="https://github.com/divyam2605/IPD" title='Code' className='youtube-Subs'>
        <img src="./Assets/Images/github.png" alt="Github Logo"/>
        <p>Mood Harmony</p>
      </a>
    </div>
    </>
  );
}

export default App;
