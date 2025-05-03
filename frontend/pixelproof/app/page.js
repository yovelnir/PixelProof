'use client';
import { useState, useEffect } from 'react';
import Image from 'next/image';
import ImageUpload from './components/ImageUpload';
import ResultShowcase from './components/ResultShowcase';
import Toast from './components/Toast';
import ProgressBar from './components/ProgressBar';
import LogoDark from './assets/dark-logo.png';
import LogoLight from './assets/light-logo.png';

export default function Home() {
  const [currentImage, setCurrentImage] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [darkMode, setDarkMode] = useState(true);
  const [step, setStep] = useState("upload"); // upload, analyzing, results

  // Check system preference on initial load
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setDarkMode(isDarkMode);
    }
  }, []);

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
  };

  const resetAnalysis = () => {
    setCurrentImage(null);
    setResult(null);
    setStep("upload");
  };

  const handleImageUpload = async (image) => {
    setCurrentImage(image);
    setStep("analyzing");
    setIsLoading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('image', image);
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Simulate API request with a timeout
      setTimeout(async () => {
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        // Mock result - in a real app, this would come from your API
        const mockResult = {
          isReal: Math.random() > 0.5, // Randomly determine if image is real or fake
          confidence: Math.random() * 0.3 + 0.7, // Random confidence between 70% and 100%
          details: {
            metadata: {
              dimensions: `${Math.round(Math.random() * 1000 + 1000)}x${Math.round(Math.random() * 1000 + 1000)}`,
              format: 'JPEG',
              size: `${Math.round(Math.random() * 5000 + 500)}KB`
            },
            anomalies: Math.round(Math.random() * 10),
            inconsistencies: Math.round(Math.random() * 5),
          }
        };
        
        setResult(mockResult);
        setStep("results");
        showToast('Image analysis completed successfully!', 'success');
        setIsLoading(false);
      }, 3000);
      
    } catch (error) {
      console.error('Error:', error);
      showToast(error.message || 'An error occurred while processing the image', 'error');
      setIsLoading(false);
      setStep("upload");
    }
  };

  return (
    <main className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-800' : 'bg-gray-100'} py-4 px-6 transition-colors duration-300`}>
        <div className="container mx-auto max-w-5xl flex justify-between items-center">
          <div className="h-auto w-auto relative transition-opacity">
            <Image 
              src={darkMode ? LogoDark : LogoLight} 
              alt="PixelProof Logo" 
              className="object-contain"
              height={70}
              priority
            />
          </div>
          <button 
            onClick={toggleDarkMode}
            className={`p-2 rounded-full ${darkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-200 text-gray-700'}`}
            aria-label="Toggle dark mode"
          >
            {darkMode ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* Main Content Container */}
      <div className="container mx-auto max-w-5xl px-4 py-10 min-h-[calc(100vh-88px)]">
        {/* App Title and Description */}
        <div className="text-center mb-8">
          <h1 className={`text-3xl md:text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
            Deepfake Detection Tool
          </h1>
          <p className={`text-lg max-w-2xl mx-auto ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Upload your image and our AI will determine if it's authentic or a deepfake.
          </p>
        </div>

        {/* Main Content Area */}
        <div className={`rounded-xl shadow-lg overflow-hidden transition-all duration-300 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          {/* Step Indicator */}
          <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} py-4 px-6 flex justify-between items-center border-b ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}>
            <div className="flex items-center space-x-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-medium text-sm ${
                step === "upload" || step === "analyzing" || step === "results" 
                  ? darkMode ? 'bg-blue-500 text-white' : 'bg-blue-600 text-white' 
                  : darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-300 text-gray-500'
              }`}>
                1
              </div>
              <span className={`${darkMode ? 'text-gray-200' : 'text-gray-800'} font-medium`}>Upload</span>
            </div>
            
            <div className={`flex-1 mx-4 h-1 ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`}>
              <div className={`h-full transition-all duration-500 ${
                step === "analyzing" || step === "results" 
                  ? darkMode ? 'bg-blue-500' : 'bg-blue-600' 
                  : 'bg-transparent'
              }`} style={{ width: (step === "upload" ? "0%" : "100%") }}></div>
            </div>

            <div className="flex items-center space-x-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-medium text-sm ${
                step === "analyzing" || step === "results" 
                  ? darkMode ? 'bg-blue-500 text-white' : 'bg-blue-600 text-white' 
                  : darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-300 text-gray-500'
              }`}>
                2
              </div>
              <span className={`${darkMode ? 'text-gray-200' : 'text-gray-800'} font-medium`}>Analyze</span>
            </div>
            
            <div className={`flex-1 mx-4 h-1 ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`}>
              <div className={`h-full transition-all duration-500 ${
                step === "results" 
                  ? darkMode ? 'bg-blue-500' : 'bg-blue-600' 
                  : 'bg-transparent'
              }`} style={{ width: (step === "results" ? "100%" : "0%") }}></div>
            </div>

            <div className="flex items-center space-x-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-medium text-sm ${
                step === "results" 
                  ? darkMode ? 'bg-blue-500 text-white' : 'bg-blue-600 text-white' 
                  : darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-300 text-gray-500'
              }`}>
                3
              </div>
              <span className={`${darkMode ? 'text-gray-200' : 'text-gray-800'} font-medium`}>Results</span>
            </div>
          </div>

          {/* Content Box */}
          <div className="p-6">
            {step === "upload" && (
              <div className="animate-fade-in">
                <ImageUpload
                  onUpload={handleImageUpload}
                  buttonText="Analyze Image"
                  darkMode={darkMode}
                />
              </div>
            )}
            
            {step === "analyzing" && (
              <div className="py-12 animate-fade-in">
                <div className="text-center">
                  <div className={`animate-spin rounded-full h-16 w-16 border-4 border-t-transparent ${darkMode ? 'border-blue-400' : 'border-blue-600'} mx-auto mb-6`}></div>
                  <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>
                    Analyzing Your Image
                  </h3>
                  <p className={`mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    Our AI is processing your image to detect potential manipulation.
                  </p>
                  {uploadProgress > 0 && (
                    <div className="max-w-md mx-auto">
                      <ProgressBar
                        progress={uploadProgress}
                        message={uploadProgress === 100 ? 'Processing complete!' : 'Processing...'}
                        darkMode={darkMode}
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {step === "results" && result && (
              <div className="animate-fade-in">
                <ResultShowcase 
                  result={result} 
                  image={currentImage} 
                  darkMode={darkMode} 
                />
                <div className="mt-8 text-center">
                  <button
                    onClick={resetAnalysis}
                    className={`px-6 py-2 rounded-md ${
                      darkMode 
                        ? 'bg-blue-500 hover:bg-blue-600 text-white' 
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                    } transition-colors duration-300`}
                  >
                    Analyze Another Image
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toast Notifications */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
          darkMode={darkMode}
        />
      )}
    </main>
  );
}
