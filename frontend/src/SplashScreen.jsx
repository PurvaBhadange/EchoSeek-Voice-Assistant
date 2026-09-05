import React, { useState, useEffect, useRef } from 'react';

export default function SplashScreen({ onComplete }) {
  const [isFadingOut, setIsFadingOut] = useState(false);
  const videoRef = useRef(null);
  const hasFinishedRef = useRef(false);

  const handleFinish = () => {
    if (hasFinishedRef.current) return;
    hasFinishedRef.current = true;
    setIsFadingOut(true);
    setTimeout(() => {
      onComplete();
    }, 900); // 900ms matching CSS fade-out duration
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleEnded = () => handleFinish();
    const handleError = (e) => {
      console.warn('Splash video playback error:', e);
      handleFinish();
    };

    const handleTimeUpdate = () => {
      // Speed up the last section of the video (after 50% duration)
      if (video.duration && video.currentTime > video.duration * 0.5) {
        video.playbackRate = 2.2;
      }
    };

    video.addEventListener('ended', handleEnded);
    video.addEventListener('error', handleError);
    video.addEventListener('timeupdate', handleTimeUpdate);

    // Attempt video playback
    video.play().catch((err) => {
      console.warn('Splash video autoplay delayed or blocked:', err);
    });

    // Safety fallback timeout in case video never fires ended event
    const safetyTimer = setTimeout(() => {
      if (!hasFinishedRef.current) {
        handleFinish();
      }
    }, 10000);

    return () => {
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('error', handleError);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      clearTimeout(safetyTimer);
    };
  }, []);

  return (
    <div
      className={`splash-overlay ${isFadingOut ? 'fade-out' : ''}`}
      onClick={handleFinish}
      style={{ cursor: 'pointer' }}
    >
      <video
        ref={videoRef}
        src="/wave_hq.mp4"
        autoPlay
        muted
        playsInline
        className="splash-video"
      />
      <div style={{
        position: 'absolute',
        bottom: '3rem',
        left: '3rem',
        zIndex: 10,
        pointerEvents: 'none'
      }}>
        <div style={{
          fontSize: '0.9rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.15em',
          color: '#DFE104',
          backgroundColor: '#09090B',
          padding: '0.5rem 1rem',
          border: '2px solid #DFE104'
        }}>
          EchoSeek — Kinetic Intelligence // Click Anywhere to Skip
        </div>
      </div>
    </div>
  );
}

