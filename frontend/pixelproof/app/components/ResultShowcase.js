'use client';
import Image from 'next/image';

const ResultShowcase = ({ result, image, darkMode = false }) => {
  if (!result) return null;

  const { isReal, confidence, details } = result;
  const percentage = (confidence * 100).toFixed(1);

  return (
    <div className={`w-full transition-colors duration-300`}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Image Preview */}
        <div className={`rounded-lg overflow-hidden shadow-md ${
          darkMode ? 'bg-gray-700' : 'bg-gray-100'
        } p-4`}>
          <h3 className={`text-lg font-semibold mb-4 ${
            darkMode ? 'text-gray-200' : 'text-gray-800'
          }`}>Uploaded Image</h3>
          
          {image && (
            <div className="relative w-full h-64 md:h-80 rounded-lg overflow-hidden border shadow-sm">
              <Image
                src={URL.createObjectURL(image)}
                alt="Uploaded Image"
                fill
                className="object-contain"
              />
            </div>
          )}
          
          {details?.metadata && (
            <div className={`mt-4 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                  </svg>
                  <span>Size: {details.metadata.dimensions}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span>Format: {details.metadata.format}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                  <span>File size: {details.metadata.size}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Analysis Results */}
        <div className={`rounded-lg ${
          darkMode ? 'bg-gray-700' : 'bg-gray-100'
        } p-6 shadow-md`}>
          <h3 className={`text-lg font-semibold mb-4 ${
            darkMode ? 'text-gray-200' : 'text-gray-800'
          }`}>Analysis Results</h3>
          
          {/* Verdict */}
          <div className={`mb-6 p-4 rounded-lg ${
            isReal
              ? darkMode ? 'bg-green-900/40 border border-green-700' : 'bg-green-100 border border-green-200'
              : darkMode ? 'bg-red-900/40 border border-red-700' : 'bg-red-100 border border-red-200'
          }`}>
            <div className="flex items-center">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                isReal
                  ? darkMode ? 'bg-green-800 text-green-200' : 'bg-green-200 text-green-800'
                  : darkMode ? 'bg-red-800 text-red-200' : 'bg-red-200 text-red-800'
              }`}>
                {isReal ? (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                )}
              </div>
              <div className="ml-4">
                <h4 className={`text-xl font-bold ${
                  isReal
                    ? darkMode ? 'text-green-300' : 'text-green-800'
                    : darkMode ? 'text-red-300' : 'text-red-800'
                }`}>
                  {isReal ? 'Authentic Image' : 'Deepfake Detected'}
                </h4>
                <p className={`${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                } mt-1`}>
                  {isReal
                    ? 'This image appears to be genuine with no signs of manipulation.'
                    : 'This image shows signs of being artificially generated or manipulated.'
                  }
                </p>
              </div>
            </div>
          </div>
          
          {/* Confidence Score */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <h4 className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                Confidence Score
              </h4>
              <span className={`font-bold ${
                isReal
                  ? darkMode ? 'text-green-400' : 'text-green-600'
                  : darkMode ? 'text-red-400' : 'text-red-600'
              }`}>
                {percentage}%
              </span>
            </div>
            <div className={`w-full h-2 rounded-full ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`}>
              <div
                className={`h-full rounded-full ${
                  isReal
                    ? darkMode ? 'bg-green-500' : 'bg-green-600'
                    : darkMode ? 'bg-red-500' : 'bg-red-600'
                }`}
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
          
          {/* Analysis Details */}
          {details && (
            <div className={`rounded-lg border ${
              darkMode ? 'border-gray-600 bg-gray-800/50' : 'border-gray-200 bg-white'
            } p-4`}>
              <h4 className={`font-semibold mb-3 ${
                darkMode ? 'text-gray-200' : 'text-gray-700'
              }`}>
                Detection Details
              </h4>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                    Detected anomalies
                  </span>
                  <span className={`font-medium ${
                    darkMode ? 'text-gray-200' : 'text-gray-800'
                  }`}>
                    {details.anomalies}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                    Visual inconsistencies
                  </span>
                  <span className={`font-medium ${
                    darkMode ? 'text-gray-200' : 'text-gray-800'
                  }`}>
                    {details.inconsistencies}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                    Overall assessment
                  </span>
                  <span className={`font-medium ${
                    isReal
                      ? darkMode ? 'text-green-400' : 'text-green-600'
                      : darkMode ? 'text-red-400' : 'text-red-600'
                  }`}>
                    {isReal ? 'No significant issues' : 'Multiple suspicious patterns'}
                  </span>
                </div>

                {details.modelDetails && (
                  <>
                    <hr className={`my-3 ${darkMode ? 'border-gray-600' : 'border-gray-200'}`} />
                    
                    <h4 className={`font-semibold mb-2 ${
                      darkMode ? 'text-gray-200' : 'text-gray-700'
                    }`}>
                      Model Information
                    </h4>
                    
                    <div className="flex justify-between">
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                        AI Probability
                      </span>
                      <span className={`font-medium ${
                        darkMode ? 'text-gray-200' : 'text-gray-800'
                      }`}>
                        {(details.modelDetails.probability * 100).toFixed(2)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                        Models Used
                      </span>
                      <span className={`font-medium ${
                        darkMode ? 'text-gray-200' : 'text-gray-800'
                      }`}>
                        {details.modelDetails.modelsUsed}
                      </span>
                    </div>
                    
                    {details.modelDetails.voteDistribution !== 'N/A' && (
                      <div className="flex justify-between">
                        <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                          Vote Distribution
                        </span>
                        <span className={`font-medium ${
                          darkMode ? 'text-gray-200' : 'text-gray-800'
                        }`}>
                          {details.modelDetails.voteDistribution}
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultShowcase; 