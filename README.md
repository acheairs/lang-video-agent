# AP Lang Video Worksheet Agent

This project automatically completes AP English Language video-review worksheets.

## Features

* Extracts video titles from a worksheet PDF
* Finds matching YouTube videos
* Retrieves transcripts
* Falls back to Whisper transcription when captions are unavailable
* Uses OpenAI to generate worksheet responses
* Produces a completed DOCX assignment
* Generates a processing report

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
