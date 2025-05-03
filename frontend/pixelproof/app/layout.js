import './globals.css'

export const metadata = {
  title: 'PixelProof - Deepfake Detection',
  description: 'Advanced AI-powered image analysis to detect and generate deepfake images',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  )
}
