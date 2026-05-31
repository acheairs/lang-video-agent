* Extracts video titles from a worksheet PDF
* Finds matching YouTube videos with yt-dlp
* Retrieves transcripts through the YouTube Transcript API
* Falls back to OpenAI Whisper for audio transcription when transcripts are unavailable
* Uses the OpenAI API to generate detailed worksheet responses
* Produces a completed DOCX assignment
* Integrates with the Google Drive API to automatically create and sync editable Google Docs
* Generates a processing report with transcript availability, errors, and review flags

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
```

Run:

```bash
python main.py --assignment input/assignment.pdf
```

Outputs:

* completed_all_ap_lang_frq_video_notes.docx
* video_processing_report.json
