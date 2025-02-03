from enum import Enum
from typing import Literal, override

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class WhisperModel(str, Enum):
    TINY_EN = "tiny.en"
    TINY = "tiny"
    BASE_EN = "base.en"
    BASE = "base"
    SMALL_EN = "small.en"
    SMALL = "small"
    MEDIUM_EN = "medium.en"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"
    LARGE_V3_TURBO = "large-v3-turbo"


class WhisperSettings(BaseSettings):
    model: WhisperModel = Field(
        WhisperModel.LARGE_V3_TURBO,
        description="Name size of the Whisper model to use. The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    model_cache_dir: str | None = Field(
        None,
        description="Overriding the default model cache dir where models downloaded from the hub are saved.",
    )
    model_dir: str | None = Field(
        None,
        description="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    lan: str = Field(
        "auto",
        alias="language",
        description="Source language code, e.g. en, de, cs, or 'auto' for language detection.",
    )
    task: Literal["transcribe", "translate"] = Field(
        "transcribe", description="Transcribe or translate."
    )
    backend: Literal[
        "faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"
    ] = Field(
        "faster-whisper", description="Load only this backend for Whisper processing."
    )
    warmup_file: str | None = Field(
        None,
        description="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.",
    )


class VadConfig(BaseSettings):
    vac: bool = Field(
        False,
        description="Use VAC = voice activity controller. Recommended. Requires torch.",
    )
    vac_chunk_size: float = Field(0.04, description="VAC sample size in seconds.")
    vad: bool = Field(
        False,
        description="Use VAD = voice activity detection, with the default parameters.",
    )


class AppSettings(BaseSettings):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "DEBUG", description="Set the log level."
    )
    diarization: bool = Field(
        False, description="Whether to enable speaker diarization."
    )

    whisper: WhisperSettings
    vad: VadConfig

    buffer_trimming: Literal["sentence", "segment"] = Field(
        "segment",
        description="""Buffer trimming strategy -- trim completed sentences marked 
        with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper.
        Sentence segmenter must be installed for 'sentence' option.""",
    )
    buffer_trimming_sec: float = Field(
        15.0,
        description="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    min_chunk_size: float = Field(
        1.0,
        description="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )

    model_config = SettingsConfigDict(
        json_file=(
            "config.json",
            "debug_config.json",
        ),
        env_file=(
            ".env",
            ".env.debug",
        ),
        env_file_encoding="utf-8",
    )

    @override
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


app_config = AppSettings()  # pyright: ignore[reportCallIssue]
