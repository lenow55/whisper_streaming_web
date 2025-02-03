import argparse
import asyncio
from contextlib import asynccontextmanager
import io
from time import time
from typing import Any

import ffmpeg
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config import app_config
from src.whisper_streaming.whisper_online import (
    backend_factory,
    online_factory,
)
from src.constants import *

# Load demo HTML for the root endpoint
with open("src/web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr, tokenizer = backend_factory(app_config)
    app.state.asr = asr
    app.state.tokenizer = tokenizer

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(html)


async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process


def done_callback(futr: asyncio.Task[Any]):
    try:
        _ = futr.result()
    except Exception as exc:
        # do_something_with(exc) if you want to (like logging)
        # or just raise
        raise


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()
    print("Loading online.")
    online = online_factory(
        app_config, websocket.app.state.asr, websocket.app.state.tokenizer
    )
    print("Online loaded.")

    if app_config.diarization:
        from src.diarization.diarization_online import DiartDiarization

        diarization = DiartDiarization(SAMPLE_RATE)

    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()

        chunk_history = []  # Will store dicts: {beg, end, text, speaker}

        while True:
            try:
                elapsed_time = int(time() - beg)
                beg = time()
                chunk = await loop.run_in_executor(
                    None, ffmpeg_process.stdout.read, 32000 * elapsed_time
                )
                if (
                    not chunk
                ):  # The first chunk will be almost empty, FFmpeg is still starting up
                    chunk = await loop.run_in_executor(
                        None, ffmpeg_process.stdout.read, 4096
                    )
                    if not chunk:  # FFmpeg might have closed
                        print("FFmpeg stdout closed.")
                        break

                pcm_buffer.extend(chunk)

                if len(pcm_buffer) >= BYTES_PER_SEC:
                    # Convert int16 -> float32
                    pcm_array = (
                        np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    pcm_buffer = bytearray()
                    online.insert_audio_chunk(pcm_array)
                    beg_trans, end_trans, trans = online.process_iter()

                    if trans:
                        chunk_history.append(
                            {
                                "beg": beg_trans,
                                "end": end_trans,
                                "text": trans,
                                "speaker": "0",
                            }
                        )

                    full_transcription += trans
                    if app_config.vad.vac:
                        buffer = online.online.concatenate_tsw(
                            online.online.transcript_buffer.buffer
                        )[
                            2
                        ]  # We need to access the underlying online object to get the buffer
                    else:
                        buffer = online.concatenate_tsw(
                            online.transcript_buffer.buffer
                        )[2]
                    if (
                        buffer in full_transcription
                    ):  # With VAC, the buffer is not updated until the next chunk is processed
                        buffer = ""

                    lines = [
                        {
                            "speaker": "0",
                            "text": "",
                        }
                    ]

                    if app_config.diarization:
                        await diarization.diarize(pcm_array)
                        diarization.assign_speakers_to_chunks(chunk_history)

                    for ch in chunk_history:
                        if (
                            app_config.diarization
                            and ch["speaker"]
                            and ch["speaker"][-1] != lines[-1]["speaker"]
                        ):
                            lines.append(
                                {
                                    "speaker": ch["speaker"][-1],
                                    "text": ch["text"],
                                }
                            )
                        else:
                            lines[-1]["text"] += ch["text"]

                    response = {"lines": lines, "buffer": buffer}
                    await websocket.send_json(response)

            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                raise e
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    stdout_reader_task.add_done_callback(done_callback)

    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            # Pass them to ffmpeg via stdin
            ffmpeg_process.stdin.write(message)
            ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        del online

        if app_config.diarization:
            # Stop Diart
            diarization.close()
