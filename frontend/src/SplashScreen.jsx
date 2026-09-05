import React, { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, SkipForward } from 'lucide-react';

export default function SplashScreen({ onComplete }) {
  const [isFadingOut, setIsFadingOut] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
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

    video.addEventListener('ended', handleEnded);
    video.addEventListener('error', handleError);

    // Attempt video playback
    video.play().catch((err) => {
      console.warn('Splash video autoplay delayed or blocked:', err);
    });

    // Safety fallback timeout in case video never fires ended event
    const safetyTimer = setTimeout(() => {
      if (!hasFinishedRef.current) {
        handleFinish();
      }
    }, 12000);

    return () => {
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('error', handleError);
      clearTimeout(safetyTimer);
    };
  }, []);

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  return (
    <div className={`splash-overlay ${isFadingOut ? 'fade-out' : ''}`}>
      <video
        ref={videoRef}
        src="/wave_hq.mp4"
        autoPlay
        muted
        playsInline
        className="splash-video"
      />
      <div className="splash-controls">
        <button
          type="button"
          onClick={toggleMute}
          className="splash-control-btn"
          title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
        >
          {isMuted ? <VolumeX size={18} /> : <Volume2 size={18} />}
        </button>
        <button
          type="button"
          onClick={handleFinish}
          className="splash-control-btn skip-btn"
          title="Skip Splash Screen"
        >
          Skip <SkipForward size={14} />
        </button>
      </div>
    </div>
  );
}
