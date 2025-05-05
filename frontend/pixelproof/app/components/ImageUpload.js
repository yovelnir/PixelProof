'use client';
import { useState, useRef } from 'react';
import Image from 'next/image';

const ImageUpload = ({ onUpload, buttonText = 'Analyze Image', darkMode = false }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
  const ALLOWED_FORMATS = ['image/jpeg', 'image/png', 'image/jpg'];

  const validateFile = (file) => {
    setError('');

    if (!file) return false;

    if (!ALLOWED_FORMATS.includes(file.type)) {
      setError('Please upload a valid image file (JPEG, PNG)');
      return false;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError('Image size should be less than 5MB');
      return false;
    }

    return true;
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    
    if (!validateFile(file)) return;

    setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (!validateFile(file)) return;
    
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleSubmit = () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }
    onUpload(selectedImage);
  };

  return (
    <div className="max-w-xl mx-auto">
      <div className={`rounded-xl transition-all duration-300 ${
        isDragging
          ? darkMode 
              ? 'shadow-lg shadow-blue-700/30 border-blue-500/70 bg-blue-900/40' 
              : 'shadow-lg shadow-blue-300/30 border-blue-400/60 bg-blue-50/80'
          : darkMode 
              ? 'border-indigo-700/50 bg-gray-900/50' 
              : 'border-gray-200/70 bg-white/40'
      } border-2 border-dashed ${
        preview ? 'p-4' : 'p-8'
      } relative overflow-hidden antialiased`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      >
        {/* Background decorative elements */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-15 z-0">
          <div className={`absolute -bottom-8 -right-8 w-40 h-40 rounded-full ${
            darkMode ? 'bg-blue-500/10' : 'bg-blue-200/20'
          } blur-xl`}></div>
          <div className={`absolute -top-4 -left-4 w-32 h-32 rounded-full ${
            darkMode ? 'bg-purple-500/10' : 'bg-purple-200/20'
          } blur-xl`}></div>
        </div>
        
        {preview ? (
          <div className="relative z-10 space-y-6">
            <div className="relative w-full h-64 sm:h-72 md:h-80 rounded-lg overflow-hidden border shadow-sm">
              <Image
                src={preview}
                alt="Preview"
                fill
                className="object-contain"
              />
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-center">
              <button
                onClick={handleButtonClick}
                className={`px-4 py-2 rounded-lg flex items-center justify-center transition-colors duration-300 ${
                  darkMode
                    ? 'bg-gray-800 hover:bg-gray-700 text-gray-100 shadow-md shadow-gray-900/30 border border-gray-700 hover:brightness-110'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-800 shadow-md shadow-gray-300/30 hover:brightness-110'
                }`}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
                Change Image
              </button>
              
              <button
                onClick={handleSubmit}
                className={`px-6 py-3 rounded-lg flex-1 sm:flex-none sm:min-w-[160px] flex items-center justify-center transition-colors duration-300 ${
                  darkMode
                    ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg shadow-blue-900/30 ring-1 ring-blue-400/30 hover:brightness-110'
                    : 'bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 text-white shadow-lg shadow-blue-500/20 hover:brightness-110'
                }`}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                </svg>
                {buttonText}
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center cursor-pointer relative z-10 antialiased" onClick={handleButtonClick}>
            <div className={`w-24 h-24 mx-auto rounded-full flex items-center justify-center mb-6 transition-colors ${
              darkMode ? 'bg-gray-800/90 border border-indigo-700/50 shadow-md shadow-blue-900/20 hover:bg-gray-700/90' : 'bg-white/80 hover:bg-white/90'
            } shadow-lg`}>
              <svg className={`w-12 h-12 ${darkMode ? 'text-blue-300' : 'text-blue-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className={`text-xl font-semibold mb-3 ${
              darkMode ? 'text-blue-100' : 'text-gray-800'
            }`}>
              Upload Your Image
            </h3>
            <p className={`text-sm mb-6 max-w-md mx-auto ${
              darkMode ? 'text-gray-200' : 'text-gray-600'
            }`}>
              Drag and drop your image here, or click to browse
            </p>
            <button
              className={`px-6 py-3 rounded-lg inline-flex items-center transition-colors duration-300 ${
                darkMode
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg shadow-blue-900/30 ring-1 ring-blue-400/30 hover:brightness-110'
                  : 'bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 text-white shadow-lg shadow-blue-500/20 hover:brightness-110'
              } font-medium`}
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              {buttonText}
            </button>
            <p className={`text-xs mt-6 ${darkMode ? 'text-blue-300/80' : 'text-gray-500'}`}>
              Supported formats: JPEG, PNG (max 5MB)
            </p>
          </div>
        )}
        
        <label htmlFor="file-upload" className="sr-only">Click to upload</label>
        <input
          id="file-upload"
          type="file"
          className="hidden"
          onChange={handleImageChange}
          accept="image/png, image/jpeg, image/jpg"
          ref={fileInputRef}
          aria-label="Click to upload"
        />
      </div>
      
      {error && (
        <div className={`mt-4 p-4 rounded-lg ${
          darkMode 
            ? 'bg-red-900/50 text-red-200 border border-red-700 shadow-md shadow-red-900/20' 
            : 'bg-red-50 text-red-600 border border-red-200'
        } animate-fade-in shadow-sm`}>
          <div className="flex items-center">
            <svg className="w-5 h-5 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload; 