"""
DriveGuard AI - Alert Manager

Handles two different audio alerts:
1. Critical alarm -> alarm.wav
2. Excessive yawning warning -> yawn_warning.wav

Audio playback is non-blocking.
"""

import os
import threading
from typing import Optional

import cv2
import numpy as np


class AlarmManager:
    """Manages non-blocking audio alarms and visual HUD alerts."""

    def __init__(
        self,
        alarm_sound_path: str = "assets/alarm.wav",
        yawn_sound_path: str = "assets/yawn_warning.wav",
    ):
        self.alarm_sound_path = alarm_sound_path
        self.yawn_sound_path = yawn_sound_path

        self.is_playing = False
        self.is_yawn_warning_playing = False

        self._initialized = False
        self._lock = threading.Lock()

        try:
            import pygame

            pygame.mixer.init()
            self._pygame = pygame
            self._initialized = True

            print("[AlertManager] Audio system initialized.")

        except Exception as exc:
            self._pygame = None
            self._initialized = False

            print(
                f"[AlertManager] Audio initialization failed: {exc}"
            )

    # =========================================================
    # Critical Alarm
    # =========================================================

    def trigger_alarm(self) -> None:
        """Start the critical alarm if it is not already playing."""

        with self._lock:

            if self.is_playing:
                return

            if not self._initialized:
                return

            if not os.path.exists(self.alarm_sound_path):
                print(
                    f"[AlertManager] Alarm file not found: "
                    f"{self.alarm_sound_path}"
                )
                return

            try:

                self._pygame.mixer.music.load(
                    self.alarm_sound_path
                )

                self._pygame.mixer.music.play(-1)

                self.is_playing = True

                print("[AlertManager] CRITICAL alarm started.")

            except Exception as exc:

                self.is_playing = False

                print(
                    f"[AlertManager] Could not play critical alarm: "
                    f"{exc}"
                )

    def stop_alarm(self) -> None:
        """Stop the critical alarm."""

        with self._lock:

            if not self._initialized:
                self.is_playing = False
                return

            try:
                self._pygame.mixer.music.stop()
            except Exception:
                pass

            self.is_playing = False

    # =========================================================
    # Yawn Warning
    # =========================================================

    def trigger_yawn_warning(self) -> None:
        """Play the yawn warning once."""

        with self._lock:

            if self.is_yawn_warning_playing:
                return

            if not self._initialized:
                return

            if not os.path.exists(self.yawn_sound_path):
                print(
                    f"[AlertManager] Yawn warning file not found: "
                    f"{self.yawn_sound_path}"
                )
                return

            try:

                # Use a separate pygame Sound object so the
                # yawn warning does not interfere with the
                # looping critical alarm.
                sound = self._pygame.mixer.Sound(
                    self.yawn_sound_path
                )

                sound.play()

                self.is_yawn_warning_playing = True

                # Reset flag automatically after the sound ends.
                duration_ms = int(sound.get_length() * 1000)

                timer = threading.Timer(
                    max(duration_ms / 1000.0, 0.1),
                    self._reset_yawn_warning,
                )

                timer.daemon = True
                timer.start()

                print(
                    "[AlertManager] YAWN warning played."
                )

            except Exception as exc:

                self.is_yawn_warning_playing = False

                print(
                    f"[AlertManager] Could not play "
                    f"yawn warning: {exc}"
                )

    def _reset_yawn_warning(self) -> None:
        """Allow the next yawn warning to play."""

        with self._lock:
            self.is_yawn_warning_playing = False

    def stop_yawn_warning(self) -> None:
        """Stop/reset the yawn warning state."""

        with self._lock:
            self.is_yawn_warning_playing = False

    # =========================================================
    # Stop Everything
    # =========================================================

    def stop_all(self) -> None:
        """Stop all audio playback."""

        with self._lock:

            if self._initialized:

                try:
                    self._pygame.mixer.stop()
                except Exception:
                    pass

            self.is_playing = False
            self.is_yawn_warning_playing = False

    # =========================================================
    # HUD
    # =========================================================

    def draw_hud(
        self,
        frame: np.ndarray,
        state_info: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw alert information on a BGR OpenCV frame."""

        if frame is None:
            return frame

        if state_info is None:
            state_info = {}

        output = frame.copy()

        level = str(
            state_info.get(
                "level",
                state_info.get(
                    "fatigue_level",
                    "NORMAL",
                ),
            )
        ).upper()

        score = state_info.get(
            "score",
            state_info.get(
                "fatigue_score",
                0.0,
            ),
        )

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        warning_text = {
            "NORMAL": "STATUS: NORMAL",
            "WARNING": "WARNING: FATIGUE DETECTED",
            "HIGH_RISK": "HIGH RISK: TAKE A BREAK",
            "CRITICAL": "CRITICAL: DRIVER DROWSINESS",
        }.get(
            level,
            f"STATUS: {level}",
        )

        overlay = output.copy()

        panel_height = 90

        cv2.rectangle(
            overlay,
            (0, 0),
            (output.shape[1], panel_height),
            (0, 0, 0),
            -1,
        )

        output = cv2.addWeighted(
            overlay,
            0.55,
            output,
            0.45,
            0,
        )

        if level == "CRITICAL":

            text_color = (0, 0, 255)

        elif level == "HIGH_RISK":

            text_color = (0, 140, 255)

        elif level == "WARNING":

            text_color = (0, 255, 255)

        else:

            text_color = (0, 255, 0)

        cv2.putText(
            output,
            warning_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"Fatigue Score: {score:.1f}/100",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        reasons = state_info.get(
            "reasons",
            [],
        )

        if reasons:

            reason_text = " | ".join(
                str(reason)
                for reason in reasons[:3]
            )

            cv2.putText(
                output,
                reason_text[:100],
                (20, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output