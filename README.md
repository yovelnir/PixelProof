# PixelProof - Deepfake Detection and Generation

PixelProof is a web application that uses advanced AI models to detect and generate deepfake images. The project consists of a React frontend and a Flask backend.

## Features

- Image authenticity analysis
- Deepfake image generation
- Real-time results with confidence scores
- Modern, responsive UI
- Support for various image formats (JPEG, PNG)

## Project Structure

```
pixelproof/
├── frontend/           # Next.js frontend application
│   └── pixelproof/    
│       ├── app/       # React components and pages
│       └── public/    # Static assets
└── backend/           # Flask backend application
    ├── app.py        # Main Flask application
    └── uploads/      # Temporary storage for uploaded images
```

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip (Python package manager)

## Setup Instructions

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend/pixelproof
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the Flask server:
   ```bash
   python app.py
   ```

The backend API will be available at `http://localhost:5000`

## API Endpoints

### POST /api/analyze
Analyzes an uploaded image to determine if it's real or fake.

Request:
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

Response:
```json
{
  "isReal": boolean,
  "confidence": float
}
```

### POST /api/generate
Generates a deepfake image based on an uploaded image.

Request:
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

Response:
```json
{
  "generatedImage": string,
  "success": boolean
}
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 