# PixelProof - Deepfake Detection

PixelProof is an advanced AI-powered image analysis platform designed to detect and generate deepfake images. This project provides a user-friendly interface for analyzing images and determining their authenticity.

## Features

- Image upload and analysis
- Deepfake detection using advanced AI algorithms
- Real-time results and confidence scores
- User-friendly interface
- Responsive design

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pixelproof.git
cd pixelproof/frontend/pixelproof
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Create a `.env.local` file in the root directory and add your environment variables:
```env
NEXT_PUBLIC_API_URL=http://localhost:3001
```

4. Run the development server:
```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm test` - Run tests
- `npm run test:watch` - Run tests in watch mode
- `npm run test:coverage` - Run tests with coverage report

## Project Structure

```
pixelproof/
├── app/                 # Next.js app directory
│   ├── api/            # API routes
│   ├── components/     # React components
│   ├── lib/           # Utility functions
│   ├── styles/        # Global styles
│   └── pages/         # Page components
├── public/            # Static files
├── tests/            # Test files
└── ...config files
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Next.js team for the amazing framework
- Tailwind CSS for the utility-first CSS framework
- All contributors and maintainers
