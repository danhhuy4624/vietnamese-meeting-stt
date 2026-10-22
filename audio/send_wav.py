import json
import sys
import wave

import websocket


FRAME_SAMPLES = 320  # 20 ms at 16 kHz


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "test-vi.wav"
    ws = None

    try:
        with wave.open(wav_path, "rb") as audio:
            if audio.getnchannels() != 1:
                raise ValueError("Audio must be mono")

            if audio.getframerate() != 16000:
                raise ValueError("Audio must use a 16000 Hz sample rate")

            if audio.getsampwidth() != 2:
                raise ValueError("Audio must use 16-bit PCM encoding")

            print(f"Audio file: {wav_path}")
            print(f"Duration: {audio.getnframes() / audio.getframerate():.2f} seconds")

            ws = websocket.create_connection(
                "ws://localhost:8000/ws/transcribe",
                timeout=120,
            )

            ready_message = ws.recv()

            if not ready_message:
                print("The server closed the connection before becoming ready.")
                return

            print("Server:", ready_message)
            print("Sending audio...")

            while True:
                chunk = audio.readframes(FRAME_SAMPLES)

                if not chunk:
                    break

                ws.send_binary(chunk)

        # Process any speech remaining in the VAD buffer.
        ws.send("flush")

        print("Audio sent.")
        print("\nTranscript:")

        ws.settimeout(120)

        while True:
            raw_message = ws.recv()

            if not raw_message:
                print(
                    "\nThe server closed the WebSocket. "
                    "Check the meeting-stt terminal for its traceback."
                )
                break

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                print(f"Unexpected server message: {raw_message!r}")
                continue

            message_type = message.get("type")

            if message_type == "processing":
                print("Processing audio...")

            elif message_type == "transcript":
                segment = message["segment"]

                print(
                    f'[{segment["started_at_ms"] / 1000:.2f}s'
                    f'–{segment["ended_at_ms"] / 1000:.2f}s] '
                    f'{segment["text"]}'
                )

                print(
                    f'Language: {segment["language"]}, '
                    f'confidence: {segment["confidence"]:.0%}, '
                    f'model: {segment["model"]}'
                )

            elif message_type == "error":
                print("Server error:", message.get("message", "Unknown error"))
                break

    except FileNotFoundError:
        print(f"Audio file not found: {wav_path}")

    except websocket.WebSocketTimeoutException:
        print("\nTimed out waiting for another transcript.")

    except websocket.WebSocketConnectionClosedException:
        print(
            "\nThe server closed the connection unexpectedly. "
            "Check the meeting-stt terminal for the actual error."
        )

    except Exception as error:
        print(f"\nError: {type(error).__name__}: {error}")

    finally:
        if ws is not None:
            ws.close()


if __name__ == "__main__":
    main()
