import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert, Activity, Eye, AlertTriangle,
  Volume2, VolumeX, Settings, Zap, Lock
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

// Controlled Web Audio API Synthesizer Singletons (Decoupled Alarms)
let audioCtx = null;

let criticalOscillator = null;
let criticalGain = null;

let yawnOscillator = null;
let yawnGain = null;

const startCriticalAlarm = () => {
  if (criticalOscillator) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    criticalOscillator = audioCtx.createOscillator();
    criticalGain = audioCtx.createGain();
    criticalOscillator.type = 'sawtooth';
    criticalOscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // 880Hz sharp tone
    criticalGain.gain.setValueAtTime(0.35, audioCtx.currentTime);
    criticalOscillator.connect(criticalGain);
    criticalGain.connect(audioCtx.destination);
    criticalOscillator.start();
  } catch (err) {
    console.warn("Critical audio error:", err);
  }
};

const stopCriticalAlarm = () => {
  if (criticalOscillator) {
    try {
      criticalOscillator.stop();
      criticalOscillator.disconnect();
    } catch (e) { }
    criticalOscillator = null;
    criticalGain = null;
  }
};

const stopYawnAlarm = () => {
  if (yawnOscillator) {
    try {
      yawnOscillator.stop();
      yawnOscillator.disconnect();
    } catch (e) { }
    yawnOscillator = null;
    yawnGain = null;
  }
};

const testSpeaker = async () => {
  console.log("[AUDIO TEST] click received");
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    console.log("[AUDIO TEST] context state before =", audioCtx.state);
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }
    console.log("[AUDIO TEST] context state after resume =", audioCtx.state);

    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(800, now);

    gain.gain.setValueAtTime(0.0, now);
    gain.gain.linearRampToValueAtTime(0.4, now + 0.05);
    gain.gain.setValueAtTime(0.4, now + 0.45);
    gain.gain.linearRampToValueAtTime(0.0, now + 0.5);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start(now);
    console.log("[AUDIO TEST] oscillator started");

    osc.stop(now + 0.5);
    setTimeout(() => {
      console.log("[AUDIO TEST] oscillator stopped");
    }, 500);
  } catch (err) {
    console.error("[AUDIO TEST] Error:", err);
  }
};

const playYawnAlert = async () => {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    console.log("[YAWN AUDIO] playYawnAlert triggered. AudioContext state before:", audioCtx.state);
    if (audioCtx.state === 'suspended') {
      console.log("[YAWN AUDIO] calling resume...");
      try {
        await audioCtx.resume();
      } catch (e) {
        console.warn("[YAWN AUDIO] resume failed:", e);
      }
      console.log("[YAWN AUDIO] AudioContext state after resume:", audioCtx.state);
    }

    stopYawnAlarm();

    console.log("[YAWN AUDIO] creating oscillator");
    const now = audioCtx.currentTime;
    yawnOscillator = audioCtx.createOscillator();
    yawnGain = audioCtx.createGain();

    yawnOscillator.type = 'sine';
    yawnOscillator.frequency.setValueAtTime(587.33, now); // D5 pitch
    yawnOscillator.frequency.linearRampToValueAtTime(440.0, now + 0.3); // Pitch dip to A4

    // Smooth gain envelope using linearRamp (0 -> 0.35 -> 0 over 2.0s)
    yawnGain.gain.setValueAtTime(0.0, now);
    yawnGain.gain.linearRampToValueAtTime(0.35, now + 0.05);
    yawnGain.gain.setValueAtTime(0.35, now + 1.8);
    yawnGain.gain.linearRampToValueAtTime(0.0, now + 2.0);

    yawnOscillator.connect(yawnGain);
    yawnGain.connect(audioCtx.destination);

    yawnOscillator.start(now);
    console.log("[YAWN AUDIO] oscillator started successfully");

    yawnOscillator.stop(now + 2.0);
    setTimeout(() => {
      console.log("[YAWN AUDIO] oscillator stopped");
      stopYawnAlarm();
    }, 2000);
  } catch (err) {
    console.error("[YAWN AUDIO] Error playing yawn alert:", err);
  }
};

export default function App() {
  // Feed Connection & Monitoring State (Runs internally automatically)
  const [monitoringActive] = useState(true);
  const [wsStatus, setWsStatus] = useState('DISCONNECTED'); // 'CONNECTING' | 'WS CONNECTED' | 'DISCONNECTED' | 'ERROR'
  const [wsUrl] = useState('ws://localhost:8000/ws/telemetry');
  const [cameraActive, setCameraActive] = useState(false);

  // Real-Time Safety Metrics (Python Backend Telemetry)
  const [ear, setEar] = useState(0.29);
  const [earLeft, setEarLeft] = useState(0.29);
  const [earRight, setEarRight] = useState(0.29);
  const [mar, setMar] = useState(0.18);
  const [eyeState, setEyeState] = useState('NORMAL');
  const [closedEyeDuration, setClosedEyeDuration] = useState(0.0);
  const [alertnessScore, setAlertnessScore] = useState(100);
  const [consecutiveLowEarFrames, setConsecutiveLowEarFrames] = useState(0);
  const [yawnCount2Min, setYawnCount2Min] = useState(0);
  const [perclos, setPerclos] = useState(0.0);
  const [headPitch, setHeadPitch] = useState(0.0);
  const [headYaw, setHeadYaw] = useState(0.0);
  const [headRoll, setHeadRoll] = useState(0.0);
  const [fatigueScore, setFatigueScore] = useState(0.0);
  const [fatigueLevel, setFatigueLevel] = useState('NORMAL');
  const [gruClass, setGruClass] = useState('NORMAL');
  const [gruConfidence, setGruConfidence] = useState(0.0);
  const [faceDetected, setFaceDetected] = useState(false);

  // Thresholds & User Audio Preferences
  const [earThreshold, setEarThreshold] = useState(0.22);
  const [marThreshold, setMarThreshold] = useState(0.60);
  const [frameThreshold, setFrameThreshold] = useState(22); // ~1.5s at 15 FPS
  const [audioMuted, setAudioMuted] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [showSettingsDrawer, setShowSettingsDrawer] = useState(false);

  // Active Alert Flags
  const [isMicrosleep, setIsMicrosleep] = useState(false);
  const [isYawnWarning, setIsYawnWarning] = useState(false);

  // Emergency / SOS Escalation State
  const [sosState, setSosState] = useState('IDLE'); // 'IDLE' | 'COUNTDOWN' | 'ACTIVATED' | 'CANCELLED'
  const [sosCountdown, setSosCountdown] = useState(10);
  const [sosEvent, setSosEvent] = useState(null);

  const sosEpisodeActiveRef = useRef(false);
  const sosTimerRef = useRef(null);
  const sosCountdownValRef = useRef(10);

  // Edge-Trigger Yawn Alarm References
  const prevYawnWarnRef = useRef(false);
  const yawnTimeoutRef = useRef(null);
  const ignoreStaleYawn3Ref = useRef(false);

  // Time Series Telemetry History Log (Bounded: 60 points)
  const [telemetryHistory, setTelemetryHistory] = useState([]);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const frameCanvasRef = useRef(null);
  const wsRef = useRef(null);

  // Internal Camera Lifecycle
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 20 },
          facingMode: "user",
        },
        audio: false,
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (error) {
      console.error("Camera access failed:", error);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  // Web Audio Activation & Audio Context Unlocking
  const enableAudio = async () => {
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      console.log("[YAWN AUDIO] AudioContext state before resume:", audioCtx.state);
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      console.log("[YAWN AUDIO] AudioContext state after resume:", audioCtx.state);
      setAudioEnabled(true);
      await testSpeaker();
    } catch (error) {
      console.error("[YAWN AUDIO] Audio activation failed:", error);
    }
  };

  // Auto-unlock AudioContext on first user interaction anywhere on page
  useEffect(() => {
    const unlockAudio = async () => {
      try {
        if (!audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume();
          console.log("[YAWN AUDIO] AudioContext unlocked via user gesture, state:", audioCtx.state);
        }
        setAudioEnabled(true);
      } catch (e) {}
    };

    window.addEventListener('click', unlockAudio);
    window.addEventListener('keydown', unlockAudio);
    window.addEventListener('pointerdown', unlockAudio);

    return () => {
      window.removeEventListener('click', unlockAudio);
      window.removeEventListener('keydown', unlockAudio);
      window.removeEventListener('pointerdown', unlockAudio);
    };
  }, []);

  // 1. Critical Alarm Lifecycle — ONLY triggers on genuine microsleep (sustained eye closure >= 1.5s)
  useEffect(() => {
    const isCritical = isMicrosleep;
    if (isCritical && audioEnabled && !audioMuted) {
      startCriticalAlarm();
    } else {
      stopCriticalAlarm();
    }
    return () => stopCriticalAlarm();
  }, [isMicrosleep, audioEnabled, audioMuted]);

  // 2. Yawn Alarm Lifecycle (Edge-Triggered for EXACTLY 2 SECONDS, then resets count to 0)
  useEffect(() => {
    const currentYawnWarn = (yawnCount2Min >= 3 || isYawnWarning);

    if (currentYawnWarn) {
      // Edge Transition: false -> true (Newly entered 3+ yawns condition)
      if (!prevYawnWarnRef.current) {
        prevYawnWarnRef.current = true;
        console.log("[YAWN AUDIO] Yawn count reached 3 -> playYawnAlert triggered. yawnCount =", yawnCount2Min);
        console.log("[YAWN AUDIO] audioMuted =", audioMuted, "audioEnabled =", audioEnabled);

        if (!audioMuted) {
          playYawnAlert();
        }

        if (yawnTimeoutRef.current) clearTimeout(yawnTimeoutRef.current);
        yawnTimeoutRef.current = setTimeout(() => {
          console.log("[YAWN AUDIO] 3rd yawn alert finished -> resetting count to 0 and re-arming alarm.");
          setYawnCount2Min(0);
          setIsYawnWarning(false);
          prevYawnWarnRef.current = false;
          ignoreStaleYawn3Ref.current = true;
          yawnTimeoutRef.current = null;
        }, 2000);
      }
    } else {
      if (prevYawnWarnRef.current && yawnCount2Min === 0) {
        console.log("[YAWN AUDIO] Yawn alarm re-armed");
        prevYawnWarnRef.current = false;
      }
    }

    return () => {
      if (yawnTimeoutRef.current) {
        clearTimeout(yawnTimeoutRef.current);
        yawnTimeoutRef.current = null;
      }
    };
  }, [yawnCount2Min, isYawnWarning, audioMuted, audioEnabled]);

  // 3. Emergency SOS Escalation Workflow Lifecycle
  useEffect(() => {
    const isCritical = isMicrosleep || fatigueLevel === 'CRITICAL' || fatigueScore >= 75;

    if (isCritical) {
      if (!sosEpisodeActiveRef.current) {
        sosEpisodeActiveRef.current = true;
        setSosState('COUNTDOWN');
        sosCountdownValRef.current = 10;
        setSosCountdown(10);

        if (sosTimerRef.current) clearInterval(sosTimerRef.current);
        sosTimerRef.current = setInterval(() => {
          sosCountdownValRef.current -= 1;
          setSosCountdown(sosCountdownValRef.current);

          if (sosCountdownValRef.current <= 0) {
            clearInterval(sosTimerRef.current);
            sosTimerRef.current = null;
            setSosState('ACTIVATED');

            const now = new Date();
            const dateStr = now.toISOString().slice(0, 10).replace(/-/g, '');
            const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '');
            const eventId = `DG-${dateStr}-${timeStr}`;

            const eventData = {
              eventId,
              timestamp: now.toLocaleTimeString(),
              fatigueScore,
              alertness: alertnessScore,
              riskLevel: fatigueLevel,
              reason: isMicrosleep ? "Prolonged Eye Closure / Microsleep" : "Critical Fatigue Index Exceeded",
              yawnCount: yawnCount2Min,
              perclos,
              ear,
              headPose: `P:${headPitch}° Y:${headYaw}° R:${headRoll}°`,
              gruState: `${gruClass} (${(gruConfidence * 100).toFixed(0)}%)`,
              status: "ACTIVATED"
            };
            setSosEvent(eventData);

            try {
              fetch('http://localhost:8000/api/sos/events', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(eventData)
              }).catch(() => {});
            } catch (e) {}
          }
        }, 1000);
      }
    } else {
      // Automatic Recovery: if driver recovers from CRITICAL, stop countdown
      if (sosState === 'COUNTDOWN') {
        if (sosTimerRef.current) {
          clearInterval(sosTimerRef.current);
          sosTimerRef.current = null;
        }
        setSosState('IDLE');
        setSosCountdown(10);
      }
      sosEpisodeActiveRef.current = false;
    }
  }, [isMicrosleep, fatigueLevel, fatigueScore, alertnessScore, yawnCount2Min, perclos, ear, headPitch, headYaw, headRoll, gruClass, gruConfidence, sosState]);

  const cancelSos = () => {
    if (sosTimerRef.current) {
      clearInterval(sosTimerRef.current);
      sosTimerRef.current = null;
    }
    setSosState('CANCELLED');
  };

  // WebSocket Ingestion Loop
  useEffect(() => {
    if (!monitoringActive) {
      setWsStatus('DISCONNECTED');
      return;
    }

    setWsStatus('CONNECTING');
    let socket;
    try {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        wsRef.current = socket;
        setWsStatus('WS CONNECTED');
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status') return;

          if (data.ear !== undefined) setEar(data.ear);
          if (data.ear_left !== undefined) setEarLeft(data.ear_left);
          if (data.ear_right !== undefined) setEarRight(data.ear_right);
          if (data.mar !== undefined) setMar(data.mar);
          if (data.eye_state !== undefined) setEyeState(data.eye_state);
          if (data.closed_eye_duration !== undefined) setClosedEyeDuration(data.closed_eye_duration);
          if (data.alertness !== undefined) setAlertnessScore(data.alertness);
          if (data.consecutive_frames !== undefined) setConsecutiveLowEarFrames(data.consecutive_frames);

          if (data.microsleep !== undefined) {
            setIsMicrosleep(data.microsleep);
          }

          if (data.yawn_count !== undefined) {
            if (data.yawn_count < 3) {
              ignoreStaleYawn3Ref.current = false;
              setYawnCount2Min(data.yawn_count);
            } else if (data.yawn_count >= 3) {
              if (!ignoreStaleYawn3Ref.current) {
                setYawnCount2Min(data.yawn_count);
              }
            }
          }

          const isYawnActive = (data.yawn_warning !== undefined) ? data.yawn_warning : ((data.yawn_count || 0) >= 3);
          if (!ignoreStaleYawn3Ref.current) {
            setIsYawnWarning(isYawnActive);
          }

          if (data.face_detected !== undefined) setFaceDetected(data.face_detected);
          if (data.perclos !== undefined) setPerclos(data.perclos);
          if (data.head_pitch !== undefined) setHeadPitch(data.head_pitch);
          if (data.head_yaw !== undefined) setHeadYaw(data.head_yaw);
          if (data.head_roll !== undefined) setHeadRoll(data.head_roll);
          if (data.fatigue_score !== undefined) setFatigueScore(data.fatigue_score);
          if (data.fatigue_level !== undefined) setFatigueLevel(data.fatigue_level);
          if (data.gru_class !== undefined) setGruClass(data.gru_class);
          if (data.gru_confidence !== undefined) setGruConfidence(data.gru_confidence);

          const timestamp = new Date().toLocaleTimeString().split(' ')[0];
          setTelemetryHistory(prev => [...prev.slice(-59), {
            time: timestamp,
            EAR: data.ear || 0,
            MAR: data.mar || 0,
            Threshold: earThreshold
          }]);
        } catch (err) {
          console.error("WS JSON parse error:", err);
        }
      };

      socket.onclose = () => {
        setWsStatus('DISCONNECTED');
      };

      socket.onerror = () => {
        setWsStatus('ERROR');
      };
    } catch (e) {
      setWsStatus('ERROR');
    }

    return () => {
      if (socket) socket.close();
      if (wsRef.current === socket) wsRef.current = null;
    };
  }, [monitoringActive, wsUrl, earThreshold]);

  // Frame Transmission Loop (~15 FPS)
  useEffect(() => {
    if (!cameraActive || !monitoringActive) return;

    const canvas = frameCanvasRef.current || document.createElement("canvas");
    frameCanvasRef.current = canvas;
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    let busy = false;

    const sendFrame = () => {
      const socket = wsRef.current;
      const video = videoRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN || !video || video.readyState < 2 || busy) return;

      busy = true;
      ctx.drawImage(video, 0, 0, 640, 480);
      canvas.toBlob((blob) => {
        if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(blob);
        }
        busy = false;
      }, "image/jpeg", 0.55);
    };

    const interval = setInterval(sendFrame, 66); // ~15 FPS
    return () => clearInterval(interval);
  }, [cameraActive, monitoringActive, wsStatus]);

  // Determine Active Alert Banner (Priority: MICROSLEEP > CRITICAL FATIGUE > EXCESSIVE YAWNING)
  const renderAlertBanner = () => {
    if (isMicrosleep) {
      return (
        <div className="bg-red-600/95 backdrop-blur-md text-white py-2.5 px-4 border-b-2 border-red-400 flex items-center justify-between animate-pulse sticky top-0 z-50 shadow-[0_0_25px_rgba(255,0,0,0.5)]">
          <div className="flex items-center space-x-3 max-w-7xl mx-auto w-full">
            <AlertTriangle className="w-6 h-6 text-yellow-300 animate-bounce flex-shrink-0" />
            <div>
              <span className="font-extrabold tracking-wider text-sm md:text-base">
                🚨 MICROSLEEP DETECTED!
              </span>
              <span className="text-xs md:text-sm text-red-100 ml-2 hidden sm:inline">
                Driver eyes closed for {closedEyeDuration}s ({consecutiveLowEarFrames} consecutive frames). Immediate hazard!
              </span>
            </div>
          </div>
          <button
            onClick={() => setIsMicrosleep(false)}
            className="text-xs bg-black/50 hover:bg-black/80 px-3 py-1 rounded text-white border border-red-300 transition"
          >
            DISMISS
          </button>
        </div>
      );
    }

    if (fatigueLevel === 'CRITICAL' || fatigueLevel === 'HIGH_RISK') {
      return (
        <div className="bg-red-700/90 backdrop-blur-md text-white py-2.5 px-4 border-b-2 border-red-400 flex items-center justify-between sticky top-0 z-50">
          <div className="flex items-center space-x-3 max-w-7xl mx-auto w-full">
            <ShieldAlert className="w-6 h-6 text-yellow-300 flex-shrink-0" />
            <div>
              <span className="font-extrabold tracking-wider text-sm md:text-base">
                🚨 CRITICAL FATIGUE ALERT
              </span>
              <span className="text-xs md:text-sm text-red-100 ml-2 hidden sm:inline">
                Composite fatigue score: {fatigueScore.toFixed(0)}/100 ({fatigueLevel}). Pull over safely!
              </span>
            </div>
          </div>
        </div>
      );
    }

    if (yawnCount2Min >= 3 || isYawnWarning) {
      return (
        <div className="bg-amber-600/90 backdrop-blur-md text-white py-2.5 px-4 border-b-2 border-amber-300 flex items-center justify-between animate-pulse sticky top-0 z-50">
          <div className="flex items-center space-x-3 max-w-7xl mx-auto w-full">
            <AlertTriangle className="w-6 h-6 text-yellow-200 flex-shrink-0" />
            <div>
              <span className="font-extrabold tracking-wider text-sm md:text-base">
                ⚠ EXCESSIVE YAWNING DETECTED
              </span>
              <span className="text-xs md:text-sm text-amber-100 ml-2 hidden sm:inline">
                3+ Yawns detected in the rolling 2-minute window ({yawnCount2Min} yawns). Take a rest break!
              </span>
            </div>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen bg-[#070A10] text-slate-100 font-sans selection:bg-cyan-500 selection:text-black">

      {/* TOP ALERT BANNER (PRIORITY: MICROSLEEP > CRITICAL FATIGUE > YAWN WARNING) */}
      {renderAlertBanner()}

      {/* HEADER BAR */}
      <header className="border-b border-cyan-900/40 bg-[#0B0F17]/90 backdrop-blur-lg sticky top-0 z-40 shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/30 text-cyan-400 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="font-bold text-lg md:text-xl tracking-tight bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
                  DRIVEGUARD AI
                </h1>
                <span className="text-[10px] uppercase tracking-widest bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800 font-mono">
                  Vision Alert v2.4
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Real-Time Driver Drowsiness & Fatigue Vision Alert System
              </p>
            </div>
          </div>

          {/* SYSTEM CONTROL BADGES */}
          <div className="flex items-center space-x-3">
            <div className="hidden lg:flex items-center space-x-2 bg-slate-900/80 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
              <Lock className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-slate-300">Local Privacy Engine</span>
            </div>

            <button
              onClick={enableAudio}
              className={`px-3 py-2 rounded-lg border text-xs font-semibold transition flex items-center space-x-1.5 ${audioEnabled
                ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800 hover:bg-emerald-900/50'
                : 'bg-amber-950/90 text-amber-300 border-amber-700 hover:bg-amber-900 animate-pulse'
                }`}
              title="Click to enable or test Web Audio alarm"
            >
              <span>{audioEnabled ? "🔊 AUDIO ARMED" : "🔊 ENABLE ALARM"}</span>
            </button>

            <button
              onClick={testSpeaker}
              className="px-2.5 py-2 rounded-lg bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-800 text-cyan-300 text-xs font-semibold transition flex items-center space-x-1"
              title="Directly test speaker audio output (500ms tone)"
            >
              <span>TEST SPEAKER</span>
            </button>

            <button
              onClick={playYawnAlert}
              className="px-2.5 py-2 rounded-lg bg-purple-950/80 hover:bg-purple-900 border border-purple-800 text-purple-300 text-xs font-semibold transition flex items-center space-x-1"
              title="Test exact 2-second yawn alert chime tone"
            >
              <span>TEST YAWN ALERT</span>
            </button>

            <button
              onClick={() => setAudioMuted(!audioMuted)}
              className={`p-2 rounded-lg border text-xs flex items-center space-x-1.5 transition ${audioMuted
                ? 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700'
                : 'bg-cyan-950/80 text-cyan-400 border-cyan-800 hover:bg-cyan-900/50'
                }`}
              title={audioMuted ? "Unmute Alarm" : "Mute Alarm"}
            >
              {audioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              <span className="hidden sm:inline">{audioMuted ? "MUTED" : "ALARM ON"}</span>
            </button>

            <button
              onClick={() => setShowSettingsDrawer(!showSettingsDrawer)}
              className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition"
              title="Threshold Settings"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* SECTION 1: TOP DASHBOARD — MAIN 3 PRIMARY CARDS ONLY */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

          {/* 1. EYE ASPECT RATIO (EAR) CARD */}
          <div className={`p-4 rounded-xl border backdrop-blur-md transition-all ${ear < earThreshold && faceDetected
            ? 'bg-red-950/40 border-red-600/70 shadow-[0_0_20px_rgba(255,0,85,0.3)]'
            : 'bg-[#0D131F] border-slate-800'
            }`}>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Eye className="w-4 h-4 text-cyan-400" /> Eye Aspect Ratio (EAR)
              </span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-mono">
                Thresh: &lt; {earThreshold}
              </span>
            </div>
            <div className="flex items-baseline justify-between mt-1">
              <div>
                <span className={`text-3xl font-extrabold tracking-tight font-mono ${ear < earThreshold && faceDetected ? 'text-red-400' : 'text-cyan-400'}`}>
                  {faceDetected ? ear.toFixed(3) : '0.000'}
                </span>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  L: {faceDetected ? earLeft.toFixed(3) : '0.000'} | R: {faceDetected ? earRight.toFixed(3) : '0.000'}
                </div>
              </div>
              <div className="text-right">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded block ${!faceDetected
                  ? 'bg-slate-900 text-slate-400 border border-slate-800'
                  : (ear < earThreshold ? 'bg-red-900/90 text-red-200 border border-red-700' : 'bg-emerald-950 text-emerald-400 border border-emerald-800')
                  }`}>
                  {faceDetected ? eyeState : 'NO FACE'}
                </span>
                <span className="text-[10px] text-slate-400 font-mono mt-1 block">
                  Closed: {closedEyeDuration}s ({consecutiveLowEarFrames}f)
                </span>
              </div>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div
                className={`h-full transition-all duration-300 ${ear < earThreshold && faceDetected ? 'bg-red-500' : 'bg-cyan-400'}`}
                style={{ width: `${faceDetected ? Math.min(100, (ear / 0.4) * 100) : 0}%` }}
              />
            </div>
          </div>

          {/* 2. MOUTH ASPECT RATIO (MAR) & YAWN CARD */}
          <div className={`p-4 rounded-xl border backdrop-blur-md transition-all ${mar > marThreshold || yawnCount2Min >= 3
            ? 'bg-amber-950/40 border-amber-600/70 shadow-[0_0_20px_rgba(255,184,0,0.25)]'
            : 'bg-[#0D131F] border-slate-800'
            }`}>
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-amber-400" /> Mouth Aspect Ratio (MAR)
              </span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-mono">
                Thresh: &gt; {marThreshold}
              </span>
            </div>
            <div className="flex items-baseline justify-between mt-1">
              <div>
                <span className={`text-3xl font-extrabold tracking-tight font-mono ${mar > marThreshold ? 'text-amber-400' : 'text-slate-200'}`}>
                  {faceDetected ? mar.toFixed(3) : '0.000'}
                </span>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  Yawns (Rolling 2m): <span className={`font-bold ${yawnCount2Min >= 3 ? 'text-amber-400' : 'text-slate-200'}`}>{yawnCount2Min}</span>
                </div>
              </div>
              <div className="text-right">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded block ${yawnCount2Min >= 3
                  ? 'bg-amber-950 text-amber-300 border border-amber-800 animate-pulse'
                  : (mar > marThreshold ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-slate-900 text-slate-400')
                  }`}>
                  {yawnCount2Min >= 3 ? '3+ YAWNS (WARN)' : (mar > marThreshold ? 'YAWNING' : 'NORMAL')}
                </span>
              </div>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div
                className={`h-full transition-all duration-300 ${yawnCount2Min >= 3 || mar > marThreshold ? 'bg-amber-400' : 'bg-emerald-500'}`}
                style={{ width: `${faceDetected ? Math.min(100, (mar / 0.8) * 100) : 0}%` }}
              />
            </div>
          </div>

          {/* 3. DRIVER ALERTNESS INDEX CARD */}
          <div className="p-4 rounded-xl border border-slate-800 bg-[#0D131F] backdrop-blur-md">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
              <span className="font-semibold uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-emerald-400" /> Driver Alertness Index
              </span>
              <span className="text-[10px] text-slate-400 font-mono">PERCLOS: {perclos.toFixed(1)}%</span>
            </div>
            <div className="flex items-baseline justify-between mt-1">
              <div>
                <span className={`text-3xl font-extrabold tracking-tight font-mono ${alertnessScore > 75 ? 'text-emerald-400' : (alertnessScore > 45 ? 'text-amber-400' : 'text-red-500')
                  }`}>
                  {alertnessScore.toFixed(0)}%
                </span>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  Fatigue Score: {fatigueScore.toFixed(0)}/100
                </div>
              </div>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${fatigueLevel === 'CRITICAL' || fatigueLevel === 'HIGH_RISK'
                ? 'bg-red-950 text-red-300 border border-red-800'
                : (fatigueLevel === 'WARNING' ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-emerald-950 text-emerald-300 border border-emerald-800')
                }`}>
                {fatigueLevel}
              </span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full mt-3 overflow-hidden border border-slate-800">
              <div
                className={`h-full transition-all duration-300 ${alertnessScore > 75 ? 'bg-emerald-400' : (alertnessScore > 45 ? 'bg-amber-400' : 'bg-red-500')
                  }`}
                style={{ width: `${alertnessScore}%` }}
              />
            </div>
          </div>

        </div>

        {/* MIDDLE SECTION: WEBCAM DRIVER VISION HUD + TELEMETRY GRAPH */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* LIVE DRIVER CAMERA HUD (5 COLUMNS) */}
          <div className="lg:col-span-5 bg-[#0D131F] border border-cyan-900/30 rounded-xl p-4 flex flex-col justify-between shadow-xl">
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-2.5 h-2.5 rounded-full ${wsStatus === 'WS CONNECTED' ? 'bg-cyan-400 animate-ping' : 'bg-slate-600'}`} />
                  <h3 className="font-bold text-sm uppercase tracking-wider text-cyan-300">
                    Live Driver Camera HUD
                  </h3>
                </div>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${wsStatus === 'WS CONNECTED' ? 'text-emerald-400 bg-emerald-950/60 border-emerald-900' : 'text-red-400 bg-red-950/60 border-red-900'}`}>
                  {wsStatus === 'WS CONNECTED' ? 'AI PIPELINE ONLINE' : 'AI PIPELINE OFFLINE'}
                </span>
              </div>
            </div>

            {/* REAL WEBCAM VIDEO DISPLAY */}
            <div className="relative flex items-center justify-center mb-3 bg-slate-950/90 rounded-lg border border-slate-800 p-2 overflow-hidden min-h-[260px]">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full max-w-[420px] h-[260px] object-cover rounded-lg transform -scale-x-100"
              />

              <div className="absolute top-4 left-4 text-[10px] font-mono bg-black/80 backdrop-blur-sm px-2.5 py-1 rounded border border-slate-700 text-slate-200">
                {cameraActive ? (
                  <span className="text-emerald-400 font-bold">● WEBCAM ACTIVE (640x480)</span>
                ) : (
                  <span className="text-red-400 font-bold">● WEBCAM PAUSED</span>
                )}
              </div>

              {!faceDetected && cameraActive && (
                <div className="absolute bottom-4 left-4 right-4 bg-black/80 backdrop-blur-sm py-1.5 px-3 rounded border border-amber-500/50 text-amber-300 text-xs text-center font-mono">
                  SEARCHING FOR DRIVER FACE...
                </div>
              )}
            </div>

            {/* REAL AI DETECTION INDICATORS (FACE, LEFT EYE, RIGHT EYE, MOUTH, GRU, HEAD POSE) */}
            <div className="grid grid-cols-2 gap-2 mt-1 pt-3 border-t border-slate-800 text-xs">
              <div className="bg-slate-900/70 p-2 rounded border border-slate-800">
                <span className="text-slate-400 text-[10px] block uppercase font-mono">Face Detection</span>
                <span className={`font-bold font-mono ${faceDetected ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {faceDetected ? 'FACE TRACKING OK' : 'NO FACE DETECTED'}
                </span>
              </div>
              <div className="bg-slate-900/70 p-2 rounded border border-slate-800">
                <span className="text-slate-400 text-[10px] block uppercase font-mono">Left Eye (EAR)</span>
                <span className={`font-bold font-mono ${faceDetected && earLeft >= earThreshold ? 'text-cyan-300' : (faceDetected ? 'text-red-400' : 'text-slate-500')}`}>
                  {faceDetected ? earLeft.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="bg-slate-900/70 p-2 rounded border border-slate-800">
                <span className="text-slate-400 text-[10px] block uppercase font-mono">Right Eye (EAR)</span>
                <span className={`font-bold font-mono ${faceDetected && earRight >= earThreshold ? 'text-cyan-300' : (faceDetected ? 'text-red-400' : 'text-slate-500')}`}>
                  {faceDetected ? earRight.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="bg-slate-900/70 p-2 rounded border border-slate-800">
                <span className="text-slate-400 text-[10px] block uppercase font-mono">Mouth (MAR)</span>
                <span className={`font-bold font-mono ${faceDetected && mar > marThreshold ? 'text-amber-400' : (faceDetected ? 'text-emerald-400' : 'text-slate-500')}`}>
                  {faceDetected ? mar.toFixed(3) : 'N/A'}
                </span>
              </div>
              <div className="bg-slate-900/70 p-2 rounded border border-slate-800">
                <span className="text-slate-400 text-[10px] block uppercase font-mono">PyTorch GRU Net</span>
                <span className="font-bold text-cyan-300 font-mono">{gruClass} — {(gruConfidence * 100).toFixed(0)}%</span>
              </div>
              <div className="bg-slate-900/70 p-2 rounded border border-slate-800">
                <span className="text-slate-400 text-[10px] block uppercase font-mono">Head Pose Orientation</span>
                <span className="font-bold text-purple-300 font-mono">P: {headPitch.toFixed(0)}° | Y: {headYaw.toFixed(0)}° | R: {headRoll.toFixed(0)}°</span>
              </div>
              <div className="col-span-2 bg-slate-900/70 p-2 rounded border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400 text-[10px] uppercase font-mono">Composite Fatigue Engine</span>
                <span className={`font-bold font-mono ${fatigueLevel === 'CRITICAL' || fatigueLevel === 'HIGH_RISK' ? 'text-red-400' : fatigueLevel === 'WARNING' ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {fatigueLevel} ({fatigueScore.toFixed(0)}/100)
                </span>
              </div>
            </div>
          </div>

          {/* REAL-TIME RECHARTS TELEMETRY GRAPH (7 COLUMNS) */}
          <div className="lg:col-span-7 bg-[#0D131F] border border-cyan-900/30 rounded-xl p-4 flex flex-col justify-between shadow-xl">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-sm uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" /> Real-Time Telemetry Stream (EAR vs MAR)
                </h3>
                <div className="flex items-center space-x-3 text-xs font-mono">
                  <div className="flex items-center space-x-1">
                    <span className="w-3 h-0.5 bg-cyan-400 inline-block" />
                    <span className="text-slate-400">EAR</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <span className="w-3 h-0.5 bg-amber-400 inline-block" />
                    <span className="text-slate-400">MAR</span>
                  </div>
                </div>
              </div>
              <p className="text-xs text-slate-400 mb-4">
                Continuous Eye Aspect Ratio (EAR) vs Mouth Aspect Ratio (MAR) logged live over WebSocket.
              </p>
            </div>

            {/* Recharts Chart */}
            <div className="h-[250px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={telemetryHistory} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="time" stroke="#475569" fontSize={10} tickLine={false} />
                  <YAxis domain={[0, 0.8]} stroke="#475569" fontSize={10} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#090D16', borderColor: '#1E293B', borderRadius: '8px', fontSize: '12px' }}
                    labelStyle={{ color: '#94A3B8' }}
                  />
                  <ReferenceLine y={earThreshold} stroke="#FF0055" strokeDasharray="3 3" label={{ value: 'EAR Limit', fill: '#FF0055', fontSize: 10 }} />

                  <Line type="monotone" dataKey="EAR" stroke="#00F0FF" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="MAR" stroke="#FFB800" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Chart Footer */}
            <div className="mt-3 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 font-mono">
              <span>Threshold Limit: EAR &lt; {earThreshold} (Microsleep Limit: 1.5s)</span>
              <span className="text-emerald-400 font-bold">{wsStatus}</span>
            </div>
          </div>

        </div>

      </main>

      {/* SETTINGS DRAWER OVERLAY */}
      {showSettingsDrawer && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex justify-end">
          <div className="w-full max-w-md bg-[#0D131F] border-l border-cyan-900/50 p-6 flex flex-col justify-between h-full shadow-2xl">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <h2 className="font-bold text-lg text-slate-100 flex items-center gap-2">
                  <Settings className="w-5 h-5 text-cyan-400" /> System Threshold Settings
                </h2>
                <button
                  onClick={() => setShowSettingsDrawer(false)}
                  className="text-slate-400 hover:text-white p-1"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6 mt-6">
                {/* EAR Threshold */}
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-2">
                    Eye Aspect Ratio (EAR) Threshold: <span className="text-cyan-400 font-mono">{earThreshold}</span>
                  </label>
                  <input
                    type="range"
                    min="0.15"
                    max="0.30"
                    step="0.01"
                    value={earThreshold}
                    onChange={(e) => setEarThreshold(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 bg-slate-900"
                  />
                  <span className="text-[10px] text-slate-500 block mt-1">Default: 0.22 (Standard MediaPipe Threshold)</span>
                </div>

                {/* MAR Threshold */}
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-2">
                    Mouth Aspect Ratio (MAR) Yawn Limit: <span className="text-amber-400 font-mono">{marThreshold}</span>
                  </label>
                  <input
                    type="range"
                    min="0.40"
                    max="0.75"
                    step="0.05"
                    value={marThreshold}
                    onChange={(e) => setMarThreshold(parseFloat(e.target.value))}
                    className="w-full accent-amber-400 bg-slate-900"
                  />
                </div>

                {/* Consecutive Frame Threshold */}
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-2">
                    Microsleep Alarm Trigger: <span className="text-purple-400 font-mono">{frameThreshold} frames</span> (~1.5s)
                  </label>
                  <input
                    type="range"
                    min="15"
                    max="90"
                    step="5"
                    value={frameThreshold}
                    onChange={(e) => setFrameThreshold(parseInt(e.target.value))}
                    className="w-full accent-purple-400 bg-slate-900"
                  />
                </div>
              </div>
            </div>

            <button
              onClick={() => setShowSettingsDrawer(false)}
              className="w-full py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-lg transition"
            >
              Apply Settings
            </button>
          </div>
        </div>
      )}

      {/* EMERGENCY / SOS WORKFLOW MODAL OVERLAY */}
      {(sosState === 'COUNTDOWN' || sosState === 'ACTIVATED') && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-lg bg-[#0E0612] border-2 border-red-500/80 rounded-2xl p-6 shadow-2xl shadow-red-950/60 animate-in fade-in zoom-in duration-200">

            {/* Modal Header */}
            <div className="flex items-center justify-between pb-4 border-b border-red-900/50">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-red-600/20 border border-red-500 flex items-center justify-center animate-pulse">
                  <ShieldAlert className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h2 className="font-extrabold text-lg text-white tracking-wide">
                    {sosState === 'COUNTDOWN' ? '🚨 CRITICAL DROWSINESS DETECTED' : '🚨 SOS ACTIVATED'}
                  </h2>
                  <span className="text-xs font-mono text-red-400 font-semibold uppercase tracking-wider block">
                    DEMO SOS / Emergency Protocol
                  </span>
                </div>
              </div>
            </div>

            {/* Modal Body: COUNTDOWN State */}
            {sosState === 'COUNTDOWN' && (
              <div className="py-5 space-y-5">
                {/* Telemetry Metrics Card */}
                <div className="grid grid-cols-3 gap-3 bg-red-950/30 border border-red-900/40 rounded-xl p-3 text-center">
                  <div>
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Fatigue Score</span>
                    <span className="text-lg font-extrabold text-red-400">{fatigueScore}/100</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Alertness</span>
                    <span className="text-lg font-extrabold text-red-400">{alertnessScore}%</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Risk Level</span>
                    <span className="text-lg font-extrabold text-red-500 animate-pulse">{fatigueLevel}</span>
                  </div>
                </div>

                <div className="text-center text-xs text-red-300 font-medium">
                  <span className="font-bold">Trigger Reason:</span> {isMicrosleep ? "Prolonged Eye Closure / Microsleep" : "Critical Fatigue Index Exceeded"}
                </div>

                {/* Big Countdown Timer Circle */}
                <div className="flex flex-col items-center justify-center py-2">
                  <span className="text-xs font-mono text-slate-400 uppercase tracking-widest mb-1">
                    Emergency protocol starting in
                  </span>
                  <div className="w-20 h-20 rounded-full border-4 border-red-500 bg-red-950/50 flex items-center justify-center shadow-lg shadow-red-500/20">
                    <span className="text-4xl font-extrabold font-mono text-white animate-pulse">
                      {sosCountdown}
                    </span>
                  </div>
                </div>

                {/* Cancel Button */}
                <button
                  onClick={cancelSos}
                  className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl border border-slate-700 transition flex items-center justify-center gap-2 shadow-lg"
                >
                  <Lock className="w-4 h-4 text-slate-400" />
                  CANCEL SOS
                </button>
              </div>
            )}

            {/* Modal Body: ACTIVATED State */}
            {sosState === 'ACTIVATED' && sosEvent && (
              <div className="py-5 space-y-5">
                <div className="bg-red-950/40 border border-red-800/60 rounded-xl p-4 space-y-2 text-xs font-mono">
                  <div className="flex justify-between border-b border-red-900/40 pb-1.5">
                    <span className="text-slate-400">Event ID:</span>
                    <span className="text-cyan-400 font-bold">{sosEvent.eventId}</span>
                  </div>
                  <div className="flex justify-between border-b border-red-900/40 pb-1.5">
                    <span className="text-slate-400">Time:</span>
                    <span className="text-slate-200">{sosEvent.timestamp}</span>
                  </div>
                  <div className="flex justify-between border-b border-red-900/40 pb-1.5">
                    <span className="text-slate-400">Fatigue Score:</span>
                    <span className="text-red-400 font-bold">{sosEvent.fatigueScore}/100</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Risk Level:</span>
                    <span className="text-red-500 font-bold">{sosEvent.riskLevel}</span>
                  </div>
                </div>

                {/* Checklist Confirmation */}
                <div className="space-y-2 text-xs font-medium text-slate-200 bg-slate-900/80 rounded-xl p-3.5 border border-slate-800">
                  <div className="flex items-center gap-2 text-emerald-400">
                    <span>✓</span> Emergency event recorded
                  </div>
                  <div className="flex items-center gap-2 text-emerald-400">
                    <span>✓</span> Driver warning issued
                  </div>
                  <div className="flex items-center gap-2 text-emerald-400">
                    <span>✓</span> Safety escalation activated
                  </div>
                </div>

                {/* Dismiss / View Incident Button */}
                <button
                  onClick={() => setSosState('IDLE')}
                  className="w-full py-3 bg-red-600 hover:bg-red-500 text-white font-bold rounded-xl transition shadow-lg shadow-red-600/30 flex items-center justify-center gap-2"
                >
                  <Activity className="w-4 h-4" />
                  VIEW INCIDENT / ACKNOWLEDGE
                </button>
              </div>
            )}

          </div>
        </div>
      )}

    </div>
  );
}