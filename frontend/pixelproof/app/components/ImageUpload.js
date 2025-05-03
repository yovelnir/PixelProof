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
      <div className={`rounded-lg border-2 ${
        isDragging
          ? darkMode ? 'border-blue-400 bg-blue-900/20' : 'border-blue-400 bg-blue-50'
          : darkMode ? 'border-gray-600 bg-gray-800/50' : 'border-gray-300 bg-gray-50/80'
      } transition-all duration-200 ${
        preview ? 'p-4' : 'p-8'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      >
        {preview ? (
          <div className="space-y-6">
            <div className="relative w-full h-64 rounded-lg overflow-hidden border shadow-sm">
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
                className={`px-4 py-2 rounded-lg flex items-center justify-center ${
                  darkMode
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                } transition-colors`}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
                Change Image
              </button>
              
              <button
                onClick={handleSubmit}
                className={`px-6 py-2 flex-1 sm:flex-none sm:min-w-[160px] rounded-lg flex items-center justify-center ${
                  darkMode
                    ? 'bg-blue-500 hover:bg-blue-600 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                } transition-colors`}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                </svg>
                {buttonText}
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center cursor-pointer" onClick={handleButtonClick}>
            <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${
              darkMode ? 'bg-gray-700' : 'bg-white'
            }`}>
              <svg className={`w-10 h-10 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
              Upload Your Image
            </h3>
            <p className={`text-sm mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Drag and drop your image here, or click to browse
            </p>
            <button
              className={`px-5 py-2 rounded-lg inline-flex items-center ${
                darkMode
                  ? 'bg-blue-500 hover:bg-blue-600 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              } transition-colors mx-auto`}
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Browse Files
            </button>
            <p className={`text-xs mt-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Supported formats: JPEG, PNG (max 5MB)
            </p>
          </div>
        )}
        
        <input
          type="file"
          className="hidden"
          onChange={handleImageChange}
          accept="image/png, image/jpeg, image/jpg"
          ref={fileInputRef}
        />
      </div>
      
      {error && (
        <div className={`mt-4 p-3 rounded-lg ${
          darkMode ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-600'
        }`}>
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