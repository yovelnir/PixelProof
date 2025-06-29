'use client';
import { useState, useEffect } from 'react';
import Image from 'next/image';
import ImageUpload from './components/ImageUpload';
import ResultShowcase from './components/ResultShowcase';
import Toast from './components/Toast';
import ProgressBar from './components/ProgressBar';
import LogoDark from './assets/dark-logo.png';
import LogoLight from './assets/light-logo.png';

// Backend API URL - update this to your actual backend URL
const API_URL = 'http://localhost:5000';

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
      // Get image dimensions
      const getImageDimensions = (file) => {
        return new Promise((resolve) => {
          const img = new window.Image();
          img.onload = () => {
            resolve({
              width: img.naturalWidth,
              height: img.naturalHeight
            });
            URL.revokeObjectURL(img.src);
          };
          img.onerror = () => {
            resolve({ width: 'unknown', height: 'unknown' });
          };
          img.src = URL.createObjectURL(file);
        });
      };

      const dimensions = await getImageDimensions(image);

      const formData = new FormData();
      formData.append('image', image);
      
      // Set up progress tracking
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 5;
        });
      }, 100);

      // Call the actual backend API
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze image');
      }
      
      const apiResult = await response.json();
      
      // Transform API response to match our frontend format
      const formattedResult = {
        // Consider both prediction and vote distribution to determine if real
        isReal: apiResult.prediction === "real" && 
                (!apiResult.vote_distribution || 
                 !apiResult.vote_distribution.includes("fake") || 
                 apiResult.vote_distribution.includes("real")),
        confidence: apiResult.confidence,
        details: {
          metadata: {
            dimensions: `${dimensions.width}x${dimensions.height}`,
            format: image.type.split('/')[1].toUpperCase(),
            size: `${Math.round(image.size / 1024)}KB`
          },
          anomalies: apiResult.prediction === "real" ? 0 : apiResult.probability * 10, 
          inconsistencies: apiResult.prediction === "real" ? 0 : Math.round(apiResult.probability * 5),
          modelDetails: {
            modelsUsed: apiResult.models_used || 1,
            voteDistribution: apiResult.vote_distribution || 'N/A',
            probability: apiResult.probability
          }
        }
      };
      
      setResult(formattedResult);
      setStep("results");
      showToast('Image analysis completed successfully!', 'success');
    } catch (error) {
      console.error('Error:', error);
      showToast(error.message || 'An error occurred while processing the image', 'error');
      setStep("upload");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className={`min-h-screen transition-colors duration-500 ${
      darkMode 
        ? 'bg-gradient-to-br from-gray-950 via-blue-950 to-gray-950 text-gray-100' 
        : 'bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 text-gray-900'
    } overflow-hidden relative`}>
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-0 right-0 w-96 h-96 rounded-full ${
          darkMode ? 'bg-blue-600/15' : 'bg-blue-200/30'
        } blur-3xl transform -translate-y-32 translate-x-32 animate-pulse-slow`}></div>
        
        <div className={`absolute bottom-0 left-0 w-80 h-80 rounded-full ${
          darkMode ? 'bg-purple-600/15' : 'bg-purple-200/30'
        } blur-3xl transform translate-y-40 -translate-x-16 animate-pulse-slower`}></div>
        
        <div className={`absolute top-1/3 left-1/4 w-72 h-72 rounded-full ${
          darkMode ? 'bg-cyan-600/15' : 'bg-cyan-200/30'
        } blur-3xl animate-float`}></div>
      </div>

      {/* Header */}
      <header className={`backdrop-blur-md ${
        darkMode ? 'bg-gray-900/90 border-b border-indigo-500/50' : 'bg-white/30 border-b border-gray-200/50'
      } py-4 px-6 transition-colors duration-300 sticky top-0 z-10`}>
        <div className="container mx-auto max-w-6xl flex justify-between items-center">
          <div className={`relative transition-all duration-300 group p-3 rounded-xl pixel-effect scanline ${
            darkMode 
              ? 'bg-indigo-500 border-2 border-blue-300 shadow-lg shadow-blue-500/50' 
              : 'bg-gradient-to-r from-white/90 via-blue-50/30 to-white/90 border border-indigo-200/30 shadow-md hover:shadow-indigo-300/50'
          }`}>
            {/* Permanently visible glow */}
            <div className={`absolute inset-0 rounded-xl ${
              darkMode ? 'bg-blue-300/30' : 'animate-glow bg-indigo-200/10'
            }`}></div>
            
            {/* Interactive hover glow effect */}
            <div className={`absolute inset-0 rounded-xl opacity-30 group-hover:opacity-100 transition-opacity duration-500 ${
              darkMode ? 'bg-blue-300/40' : 'bg-indigo-100/40'
            }`}>
              <div className={`absolute inset-0 rounded-xl ${
                darkMode 
                  ? 'bg-gradient-to-tr from-blue-300/30 via-blue-200/40 to-indigo-200/30' 
                  : 'bg-gradient-to-tr from-blue-100/0 via-indigo-300/20 to-purple-200/0'
              }`}></div>
            </div>
            
            {/* Logo shine effect */}
            <div className="absolute inset-0 rounded-xl overflow-hidden">
              <div className={`absolute h-full w-1/4 translate-y-0 -translate-x-full group-hover:translate-x-[400%] ${
                darkMode ? 'bg-white/40' : 'bg-indigo-300/20'
              } blur-md -skew-x-12 transition-transform duration-1500 ease-in-out`}></div>
            </div>
            
            <div className="relative z-10 flex items-center">
              <div className="flex-shrink-0">
                <Image 
                  src={darkMode ? LogoDark : LogoLight} 
                  alt="PixelProof Logo" 
                  className="object-contain transform group-hover:scale-105 transition-transform duration-300"
                  height={65}
                  priority
                />
              </div>
              <div className="ml-3 flex flex-col">
                <span className={`font-bold text-lg md:text-xl ${
                  darkMode 
                    ? 'text-white drop-shadow-[0_0_3px_rgba(255,255,255,1)]' 
                    : 'text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-indigo-800'
                }`}>
                  PixelProof
                </span>
                <span className={`text-xs font-medium tracking-wide ${
                  darkMode ? 'text-white font-semibold drop-shadow-[0_0_2px_rgba(255,255,255,0.8)]' : 'text-indigo-700/80'
                }`}>
                  Deepfake Detection
                </span>
              </div>
            </div>
          </div>
          
          <button 
            onClick={toggleDarkMode}
            className={`p-3 rounded-full transition-all duration-300 hover:scale-110 ${
              darkMode 
                ? 'bg-gray-700 text-yellow-300 hover:bg-gray-600 hover:text-yellow-200 shadow-md shadow-black/20 border border-yellow-400/30' 
                : 'bg-white/70 text-gray-700 hover:bg-white/90 hover:text-gray-800'
            } backdrop-blur-sm`}
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
      <div className="container mx-auto max-w-6xl px-4 sm:px-6 py-10 min-h-[calc(100vh-88px)] relative z-0">
        {/* App Title and Description */}
        <div className="text-center mb-10 max-w-3xl mx-auto animate-fade-in">
          <h1 className={`text-3xl md:text-5xl font-bold mb-6 ${
            darkMode 
              ? 'text-transparent bg-clip-text bg-gradient-to-r from-blue-200 to-indigo-200 drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]' 
              : 'text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-purple-700'
          }`}>
            Deepfake Detection Tool
          </h1>
          <p className={`text-lg md:text-xl max-w-2xl mx-auto ${
            darkMode ? 'text-blue-100' : 'text-indigo-800'
          } leading-relaxed`}>
            Upload your image and our AI will determine if it's authentic or a deepfake.
          </p>
        </div>

        {/* Main Content Area */}
        <div className={`rounded-2xl shadow-xl overflow-hidden transition-all duration-300 max-w-4xl mx-auto transform hover:scale-[1.01] ${
          darkMode 
            ? 'bg-gray-900/60 backdrop-blur-md border border-indigo-700/30 shadow-blue-900/30' 
            : 'bg-white/70 backdrop-blur-md border border-gray-200/50 shadow-indigo-200/30'
        }`}>
          {/* Step Indicator */}
          <div className={`${
            darkMode 
              ? 'bg-gradient-to-r from-gray-900/90 to-gray-800/90 border-b border-indigo-700/30' 
              : 'bg-gradient-to-r from-gray-50/90 to-white/90 border-b border-gray-200/50'
          } py-5 px-6 flex justify-between items-center`}>
            <div className="flex items-center space-x-3">
              <div className={`w-9 h-9 rounded-full flex items-center justify-center font-medium text-sm transition-all ${
                step === "upload" || step === "analyzing" || step === "results" 
                  ? darkMode 
                      ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-900/40 ring-1 ring-blue-400/30' 
                      : 'bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-lg shadow-blue-500/20' 
                  : darkMode 
                      ? 'bg-gray-700 text-gray-400' 
                      : 'bg-gray-300 text-gray-500'
              }`}>
                1
              </div>
              <span className={`${darkMode ? 'text-blue-100' : 'text-gray-800'} font-medium hidden sm:inline`}>Upload</span>
            </div>
            
            <div className={`flex-1 mx-2 sm:mx-4 h-1 rounded-full ${darkMode ? 'bg-gray-700/80' : 'bg-gray-300/50'} overflow-hidden`}>
              <div className={`h-full transition-all duration-500 rounded-full ${
                step === "analyzing" || step === "results" 
                  ? darkMode 
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-500 shadow-glow-blue' 
                      : 'bg-gradient-to-r from-blue-500 to-indigo-600' 
                  : 'bg-transparent'
              }`} style={{ width: (step === "upload" ? "0%" : "100%") }}></div>
            </div>

            <div className="flex items-center space-x-3">
              <div className={`w-9 h-9 rounded-full flex items-center justify-center font-medium text-sm transition-all ${
                step === "analyzing" || step === "results" 
                  ? darkMode 
                      ? 'bg-gradient-to-br from-indigo-500 to-indigo-600 text-white shadow-lg shadow-indigo-900/40 ring-1 ring-indigo-400/30' 
                      : 'bg-gradient-to-br from-indigo-500 to-indigo-700 text-white shadow-lg shadow-indigo-500/20' 
                  : darkMode 
                      ? 'bg-gray-700 text-gray-400' 
                      : 'bg-gray-300 text-gray-500'
              }`}>
                2
              </div>
              <span className={`${darkMode ? 'text-blue-100' : 'text-gray-800'} font-medium hidden sm:inline`}>Analyze</span>
            </div>
            
            <div className={`flex-1 mx-2 sm:mx-4 h-1 rounded-full ${darkMode ? 'bg-gray-700/80' : 'bg-gray-300/50'} overflow-hidden`}>
              <div className={`h-full transition-all duration-500 rounded-full ${
                step === "results" 
                  ? darkMode 
                      ? 'bg-gradient-to-r from-indigo-500 to-purple-500 shadow-glow-indigo' 
                      : 'bg-gradient-to-r from-indigo-500 to-purple-600' 
                  : 'bg-transparent'
              }`} style={{ width: (step === "results" ? "100%" : "0%") }}></div>
            </div>

            <div className="flex items-center space-x-3">
              <div className={`w-9 h-9 rounded-full flex items-center justify-center font-medium text-sm transition-all ${
                step === "results" 
                  ? darkMode 
                      ? 'bg-gradient-to-br from-purple-500 to-purple-600 text-white shadow-lg shadow-purple-900/40 ring-1 ring-purple-400/30' 
                      : 'bg-gradient-to-br from-purple-500 to-purple-700 text-white shadow-lg shadow-purple-500/20' 
                  : darkMode 
                      ? 'bg-gray-700 text-gray-400' 
                      : 'bg-gray-300 text-gray-500'
              }`}>
                3
              </div>
              <span className={`${darkMode ? 'text-blue-100' : 'text-gray-800'} font-medium hidden sm:inline`}>Results</span>
            </div>
          </div>

          {/* Content Box */}
          <div className="p-6 md:p-8">
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
                  <div className={`relative mx-auto mb-8 w-20 h-20`}>
                    <div className={`absolute inset-0 rounded-full ${
                      darkMode ? 'bg-blue-500/30' : 'bg-blue-100'
                    } animate-ping-slow opacity-75`}></div>
                    <div className={`animate-spin rounded-full h-20 w-20 border-4 border-t-transparent relative z-10 ${
                      darkMode ? 'border-blue-300' : 'border-blue-600'
                    }`}></div>
                  </div>
                  <h3 className={`text-xl md:text-2xl font-semibold mb-4 ${
                    darkMode ? 'text-blue-100' : 'text-blue-700'
                  }`}>
                    Analyzing Your Image
                  </h3>
                  <p className={`mb-6 max-w-lg mx-auto ${darkMode ? 'text-gray-200' : 'text-gray-600'}`}>
                    Our AI is processing your image to detect potential manipulation. This might take a moment.
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
                <div className="mt-10 text-center">
                  <button
                    onClick={resetAnalysis}
                    className={`px-6 py-3 rounded-xl ${
                      darkMode 
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg shadow-blue-900/40 ring-1 ring-blue-400/30' 
                        : 'bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 text-white shadow-lg shadow-blue-500/20'
                    } transition-all duration-300 transform hover:scale-105 font-medium`}
                  >
                    Analyze Another Image
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Footer credits - subtle at the bottom */}
        <div className="mt-16 text-center opacity-70 hover:opacity-100 transition-opacity">
          <p className={`text-xs ${darkMode ? 'text-blue-200' : 'text-blue-800/70'}`}>
            Powered by advanced AI image analysis technology
          </p>
        </div>
      </div>

      {/* Toast notifications */}
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
