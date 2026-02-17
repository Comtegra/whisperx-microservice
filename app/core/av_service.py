import io
from typing import IO, Iterator

import av
from loguru import logger


class AudioExtractor:
    """
    Extracts audio from video files and streams as MP3.
    Uses PyAV (FFmpeg bindings) for efficient audio processing.
    """

    def __init__(self, input_file_obj: IO[bytes], output_format: str = "mp3"):
        if not (hasattr(input_file_obj, "read") and hasattr(input_file_obj, "seek")):
            raise TypeError(
                "input_file_obj must be a readable and seekable file-like object"
            )
        self.input_file_obj = input_file_obj
        self.output_format = output_format

    def stream_mp3_bytes(self) -> Iterator[bytes]:
        """
        Stream audio as MP3 bytes from video file.
        Yields chunks of MP3 data as they are encoded.
        """
        input_container = None
        output_container = None
        output_buffer = io.BytesIO()

        try:
            self.input_file_obj.seek(0)
            input_container = av.open(self.input_file_obj, "r")

            input_stream = next(
                (s for s in input_container.streams if s.type == "audio"), None
            )

            if input_stream is None:
                raise ValueError("No audio stream found in the input file")

            output_container = av.open(
                output_buffer, mode="w", format=self.output_format
            )
            sample_rate = input_stream.codec_context.sample_rate
            layout = input_stream.codec_context.layout or "stereo"

            output_stream = output_container.add_stream(
                self.output_format, rate=sample_rate
            )
            output_stream.codec_context.layout = layout

            for packet in input_container.demux(input_stream):
                if packet.dts is None:
                    continue

                for frame in packet.decode():
                    frame.pts = None
                    for output_packet in output_stream.encode(frame):
                        output_container.mux(output_packet)
                        output_buffer.seek(0)
                        chunk = output_buffer.read()
                        output_buffer.seek(0)
                        output_buffer.truncate()
                        if chunk:
                            yield chunk

            for output_packet in output_stream.encode(None):
                output_container.mux(output_packet)
                output_buffer.seek(0)
                chunk = output_buffer.read()
                output_buffer.seek(0)
                output_buffer.truncate()
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            raise RuntimeError(f"An error occurred during audio extraction: {e}") from e
        finally:
            if input_container is not None:
                input_container.close()
            if output_container is not None:
                output_container.close()
            if not output_buffer.closed:
                output_buffer.close()
