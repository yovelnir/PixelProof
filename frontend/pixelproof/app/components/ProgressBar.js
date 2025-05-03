'use client';

const ProgressBar = ({ progress, message, darkMode = false }) => {
  return (
    <div className="w-full">
      <div className="relative pt-1">
        <div className="flex mb-2 items-center justify-between">
          <div>
            <span className={`text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full ${
              darkMode 
                ? 'text-blue-300 bg-gray-700' 
                : 'text-blue-600 bg-blue-200'
            }`}>
              {message}
            </span>
          </div>
          <div className="text-right">
            <span className={`text-xs font-semibold inline-block ${
              darkMode ? 'text-blue-300' : 'text-blue-600'
            }`}>
              {progress}%
            </span>
          </div>
        </div>
        <div className={`overflow-hidden h-2 mb-4 text-xs flex rounded ${
          darkMode ? 'bg-gray-700' : 'bg-blue-200'
        }`}>
          <div
            style={{ width: `${progress}%` }}
            className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
              darkMode ? 'bg-blue-500' : 'bg-blue-500'
            } transition-all duration-500`}
          />
        </div>
      </div>
    </div>
  );
};

export default ProgressBar; 